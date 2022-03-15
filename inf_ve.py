"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of VQA for submission
"""
import argparse
import json
import os
from os.path import exists
from torch.nn import functional as F

import torch
from torch.utils.data import DataLoader

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb,
                  VeDataset, VeEvalDataset,
                  ve_collate, ve_eval_collate)
from utils.misc import VE_IDX2ENT as label2ans

from apex import amp
from horovod import torch as hvd
import numpy as np
from cytoolz import concat

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, VqaEvalDataset, vqa_eval_collate)
from model.ve_MSD_feature import UniterForVisualEntailment

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import BUCKET_SIZE, IMG_DIM


def create_dataloader(img_path, txt_path, batch_size, is_train,
                      dset_cls, collate_fn, opts,model_opts):
    img_db = DetectFeatLmdb(img_path, model_opts.conf_th, model_opts.max_bb, model_opts.min_bb,
                            model_opts.num_bb, opts.compressed_db)

    txt_db = TxtTokLmdb(txt_path,  model_opts.max_txt_len if is_train else -1)
    dset = dset_cls(txt_db, img_db)
    sampler = TokenBucketSampler(dset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train,size_multiple=1)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=model_opts.n_workers, pin_memory=model_opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()

    LOGGER.info("device: {} n_gpu: {}, rank: {}, " "16-bits training: {}".format(device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(json.load(open(hps_file)))


    val_dataloader = create_dataloader(opts.val_img_db, opts.val_txt_db,
                                       opts.val_batch_size, False,
                                       VeEvalDataset, ve_eval_collate, opts,model_opts)
    test_dataloader = create_dataloader(opts.test_img_db, opts.test_txt_db,
                                        opts.val_batch_size, False,
                                        VeEvalDataset, ve_eval_collate, opts,model_opts)



    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'

    checkpoint = torch.load(ckpt_file)

    model = UniterForVisualEntailment.from_pretrained(
        f'{opts.output_dir}/log/model.json', state_dict=checkpoint, img_dim=IMG_DIM)


    model.uniter.encoder.set_early_exit_entropy(opts.early_exit_entropy)
    model.uniter.init_early_exit_pooler()
    model.to(device)

    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')
    if not args.eval_each_layer:
        for split, loader in [("val", val_dataloader),
                              ("test", test_dataloader)]:
            LOGGER.info(
                        f"validation on {split} split...")
            val_log, results, logits,labels = evaluate(model, loader, label2ans, split, eval_threshold=True)
            result_dir = f'{opts.output_dir}/results_test'
            if not exists(result_dir) and rank == 0:
                os.makedirs(result_dir)
            torch.save(logits,f'{result_dir}/logits_{opts.checkpoint}_{split}_{opts.early_exit_entropy}_all.pt')
            torch.save(labels,f'{result_dir}/lobels_{opts.checkpoint}_{split}_{opts.early_exit_entropy}_all.pt')


    if args.eval_each_layer:
        for split, loader in [("val", val_dataloader),
                              ("test", test_dataloader)]:
            for i in range(model.num_layers):
                print("layer:", i)
                LOGGER.info("\n")
                _result = evaluate(model,loader, label2ans, split,output_layer=i, eval_threshold=True)
                result_dir = f'{opts.output_dir}/results_test'





@torch.no_grad()
def evaluate(model, val_loader, label2ans,split='val', output_layer=-1, eval_threshold=True):
    LOGGER.info(f"start running {split}...")

    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0
    results = {}
    logits = []
    labels = []
    exit_layer_counter = {(i + 1): 0 for i in range(model.num_layers)}
    kls= []


    for i, batch in enumerate(val_loader):
        if output_layer >= 0:
            batch['output_layer'] = output_layer

        scoress = model(batch, compute_loss=True,output_layer=output_layer)
        eval_loss, scores = scoress[:2]
        kls.append(eval_loss[-1])

        if eval_threshold:
            exit_layer_counter[scoress[-1]] += 1

        targets = batch['targets']
        labels.append(targets)
        loss = F.binary_cross_entropy_with_logits(
            scores, targets, reduction='sum')

        val_loss += loss.item()
        tot_score += compute_score_with_logits(scores, targets).sum().item()
        answers = [label2ans[i]
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        qids = batch['qids']
        for qid, answer in zip(qids, answers):
            results[qid] = answer
        n_ex += len(qids)

        scores = scores.cpu()
        logits.append(scores)

        print(
            "\rIteration: {:>4}/{}   Loss: {:.5f} Scores: {:.5f}".format(
                i, len(val_loader.dataset),
                loss,tot_score),
            end="")
    klss = sum(kls)
    print("\nKL",klss/len(val_loader.dataset))

    if eval_threshold:
        print("Exit layer counter", exit_layer_counter)
        # LOGGER.info( exit_layer_counter)
        actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
        full_cost = len(val_loader.dataset) * model.num_layers
        print("Expected saving", actual_cost / full_cost)
        # LOGGER.info(f"Expected saving: {actual_cost / full_cost: .5f}")


    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/{split}_loss': val_loss,
               f'valid/{split}_acc': val_acc,
               }
    model.train()
    LOGGER.info(exit_layer_counter)
    LOGGER.info(f"Expected saving: {actual_cost / full_cost: .5f}")

    LOGGER.info(
                f"score: {val_acc*100:.2f}")
    return val_log, results, logits,labels


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--val_txt_db",
                        default=None, type=str,
                        help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_db",
                        default=None, type=str,
                        help="The input validation images.")
    parser.add_argument("--test_txt_db",
                        default=None, type=str,
                        help="The input test corpus. (LMDB)")
    parser.add_argument("--test_img_db",
                        default=None, type=str,
                        help="The input test images.")
    parser.add_argument("--eval_each_layer", action='store_true', help="Set this flag to evaluate each layer.")

    parser.add_argument('--compressed_db', action='store_true',help='use compressed LMDB')
    parser.add_argument("--checkpoint",default=None, type=str,help="can be the path to binary or int number (step)")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Total batch size for validation. " "(batch by tokens)")

    parser.add_argument("--early_exit_entropy", default=-1, type=float, help="Entropy threshold for early exit.")

    parser.add_argument("--output_dir", default=None, type=str,help="The output directory of the training command")

    parser.add_argument("--save_logits", action='store_true',help="Whether to save logits (for making ensemble)")

    # Prepro parameters

    # device parameters
    parser.add_argument('--fp16',action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',help="pin memory")

    args = parser.parse_args()

    main(args)
