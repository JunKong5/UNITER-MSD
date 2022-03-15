"""run inference of NLVR2 (single GPU only)"""
import argparse
import json
import os
from os.path import exists

import torch
from torch.utils.data import DataLoader
from apex import amp
from horovod import torch as hvd
from utils.logger import LOGGER
from torch.nn import functional as F

from data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  Nlvr2PairedEvalDataset, Nlvr2TripletEvalDataset,
                  nlvr2_paired_eval_collate, nlvr2_triplet_eval_collate)
from model.model import UniterConfig
from model.nlvr2_EE import (UniterForNlvr2Paired, UniterForNlvr2Triplet,
                         UniterForNlvr2PairedAttn)
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.misc import Struct
from utils.const import IMG_DIM, BUCKET_SIZE

def create_dataloader(img_path, txt_path, batch_size, is_train,
                      dset_cls, collate_fn, opts,train_opts):
    img_db = DetectFeatLmdb(img_path, train_opts.conf_th, train_opts.max_bb, train_opts.min_bb,
                            train_opts.num_bb, opts.compressed_db)
    txt_db = TxtTokLmdb(txt_path, train_opts.max_txt_len if is_train else -1)
    dset = dset_cls(txt_db, img_db, train_opts.use_img_type)
    sampler = TokenBucketSampler(dset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=train_opts.n_workers, pin_memory=train_opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)

def main(opts):
    hvd.init()
    device = torch.device("cuda")  # support single GPU only
    train_opts = Struct(json.load(open(f'{opts.train_dir}/log/hps.json')))

    if 'paired' in train_opts.model:
        EvalDatasetCls = Nlvr2PairedEvalDataset
        eval_collate_fn = nlvr2_paired_eval_collate
        if train_opts.model == 'paired':
            ModelCls = UniterForNlvr2Paired
        elif train_opts.model == 'paired-attn':
            ModelCls = UniterForNlvr2PairedAttn
        else:
            raise ValueError('unrecognized model type')
    elif train_opts.model == 'triplet':
        EvalDatasetCls = Nlvr2TripletEvalDataset
        ModelCls = UniterForNlvr2Triplet
        eval_collate_fn = nlvr2_triplet_eval_collate
    else:
        raise ValueError('unrecognized model type')



    val_dataloader = create_dataloader(opts.val_img_db, opts.val_txt_db,
                                       opts.val_batch_size, False,
                                       EvalDatasetCls, eval_collate_fn, opts,train_opts)
    test_dataloader = create_dataloader(opts.test_img_db, opts.test_txt_db,
                                        opts.val_batch_size, False,
                                        EvalDatasetCls, eval_collate_fn, opts,train_opts)


    # Prepare model
    ckpt_file = f'{opts.train_dir}/ckpt/model_step_{opts.ckpt}.pt'

    checkpoint = torch.load(ckpt_file)


    model = ModelCls.from_pretrained(f'{opts.train_dir}/log/model.json', img_dim=IMG_DIM,state_dict=checkpoint, num_answer=2)

    model.uniter.encoder.set_early_exit_entropy(opts.early_exit_entropy)

    model.init_type_embedding()
    # model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
    if not args.eval_each_layer:

        for split, loader in [('val', val_dataloader),
                              ('test', test_dataloader)]:
            log, results = evaluate(model, loader,split, eval_threshold =True)


    if args.eval_each_layer:
        each_layer_results = []
        for split, loader in [("val", val_dataloader),
                              ("test", test_dataloader)]:
            for i in range(model.num_layers):
                print("layer:", i)
                LOGGER.info("\n")
                _result = evaluate(model,loader, split,output_layer=i, eval_threshold=True)
                result_dir = f'{opts.output_dir}/results_test'

@torch.no_grad()
def evaluate(model,val_loader,split='val', output_layer=-1, eval_threshold=True):
    LOGGER.info(f"start running {split}...")
    model.eval()
    n_ex = 0
    results = []
    val_loss = 0
    tot_score = 0
    exit_layer_counter = {(i + 1): 0 for i in range(model.num_layers)}

    for i, batch in enumerate(val_loader):

        if output_layer >= 0:
            batch['output_layer'] = output_layer

        qids = batch['qids']
        targets = batch['targets']
        del batch['qids']

        scoress = model(batch, compute_loss=True,output_layer=output_layer)
        eval_loss, scores = scoress[:2]
        if eval_threshold:
            exit_layer_counter[scoress[-1]] += 1

        loss = F.cross_entropy(scores, targets, reduction='sum')

        val_loss += loss.item()
        tot_score += (scores.max(dim=-1, keepdim=False)[1] == targets
                      ).sum().item()
        answers = ['True' if i == 1 else 'False'
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        results.extend(zip(qids, answers))
        n_ex += len(qids)

        print(
            "\rIteration: {:>4}/{}   Loss: {:.5f} Scores: {:.5f}".format(
                i, len(val_loader.dataset),
                loss, tot_score),
            end="")

    if eval_threshold:
        print("Exit layer counter", exit_layer_counter)
        LOGGER.info( exit_layer_counter)
        actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
        full_cost = len(val_loader.dataset) * model.num_layers
        print("Expected saving", actual_cost / full_cost)
        LOGGER.info(f"Expected saving: {actual_cost / full_cost: .5f}")


    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/{split}_loss': val_loss,
               f'valid/{split}_acc': val_acc,
               }
    model.train()
    LOGGER.info(
                f"score: {val_acc*100:.2f}")


    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters


    parser.add_argument("--val_txt_db",default=None,  required=True,type=str,help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_db",default=None,  required=True,type=str,help="The input validation images.")
    parser.add_argument("--test_txt_db",default=None,  required=True,type=str,help="The input test corpus. (LMDB)")
    parser.add_argument("--test_img_db",default=None,  required=True,type=str,help="The input test images.")

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--val_batch_size", default=1, type=int,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', action='store_true',
                        help="fp16 inference")
    parser.add_argument("--eval_each_layer", action='store_true', help="Set this flag to evaluate each layer.")

    parser.add_argument("--early_exit_entropy", default=-1, type=float, help="Entropy threshold for early exit.")


    parser.add_argument("--train_dir", type=str, required=True,
                        help="The directory storing NLVR2 finetuning output")
    parser.add_argument("--ckpt", type=int, required=True,
                        help="specify the checkpoint to run inference")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the prediction "
                             "results will be written.")
    args = parser.parse_args()

    main(args)
