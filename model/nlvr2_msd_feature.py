"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for NLVR2 model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .model_msd_nlvr2_feature import UniterPreTrainedModel, UniterModel
from .attention import MultiheadAttention
from .model_msd_nlvr2_feature import EElayerException



class FKDLoss(nn.Module):
    def __init__(self, nBlocks, gamma, T, num_labels):
        super(FKDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss().cuda()
        self.VID_Loss = VIDLoss(num_labels)
        self.ce_loss = F.cross_entropy
        self.mse_loss = F.mse_loss
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.nBlocks = nBlocks
        self.gamma = gamma
        self.T = T
        self.num_labels = num_labels

    def forward(self, outputs, last_hidden_feature, highway_outputs,  eachlayer_hiden_feature_all, targets, soft_targets):

        if self.num_labels == 1:
            last_loss = self.mse_loss(outputs, targets)
        else:

            last_loss = self.ce_loss(outputs, targets,reduction='none').mean()

        T = self.T
        multi_losses = []
        distill_losses = []
        feature_lossses = []
        hidden_feature_losses = []

        for i in range(self.nBlocks - 1):
            if self.num_labels == 1:
                _mse = (1. - self.gamma) * self.mse_loss(highway_outputs[i], targets)
                _kld = self.kld_loss(self.log_softmax(highway_outputs[i] / T),
                                     self.softmax(soft_targets / T)) * self.gamma * T * T
                multi_losses.append(_mse)
                distill_losses.append(_kld)
            else:

                _ce = (1. - self.gamma) * self.ce_loss(highway_outputs[i], targets,reduction="none")

                _kld = self.kld_loss(self.log_softmax(highway_outputs[i] / T),
                                     self.softmax(soft_targets / T)) * self.gamma * T * T


                hidden_feature = self.mse_loss(last_hidden_feature,eachlayer_hiden_feature_all[i])*1e-6

                multi_losses.append(_ce)
                distill_losses.append(_kld)
                hidden_feature_losses.append(hidden_feature)
        m_loss = sum(multi_losses)
        d_loss = sum(distill_losses)
        hf_loss = sum(hidden_feature_losses)
        l_loss = last_loss
        loss = l_loss + d_loss + m_loss + hf_loss

        return (loss, l_loss, d_loss, m_loss)


class UniterForNlvr2Paired(UniterPreTrainedModel):

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.nlvr2_output = nn.Linear(config.hidden_size*2, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        # concat CLS of the pair
        n_pair = pooled_output.size(0) // 2
        reshaped_output = pooled_output.contiguous().view(n_pair, -1)
        answer_scores = self.nlvr2_output(reshaped_output)

        if compute_loss:
            targets = batch['targets']
            nlvr2_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return nlvr2_loss
        else:
            return answer_scores


class UniterForNlvr2Triplet(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2 (triplet format)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.nlvr2_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.nlvr2_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            nlvr2_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return nlvr2_loss
        else:
            return answer_scores


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


class UniterForNlvr2PairedAttn(UniterPreTrainedModel):
    """ Finetune UNITER for NLVR2
        (paired format with additional attention layer)
    """
    def __init__(self, config, img_dim,num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, num_answer)
        self.num_layers = config.num_hidden_layers
        self.num_labels = 2
        self.attn1 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)

        self.attn2 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)

        self.fc = nn.Sequential(
            nn.Linear(2*config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))

        self.attn_pool = AttentionPool(config.hidden_size,
                                       config.attention_probs_dropout_prob)

        self.nlvr2_output = nn.Linear(2*config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[2, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb


    def output(self, sequence_output, attn_masks):
        # separate left image and right image
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs // 2, tl * 2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs // 2, tl * 2
                                                       ).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out,
                                 key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out,
                                 key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                           ).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                            ).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        answer_scores = self.nlvr2_output(
            torch.cat([left_out, right_out], dim=-1))
        return answer_scores ,torch.cat([left_out, right_out],dim=-1)


    def forward(self, batch, compute_loss=True, output_layer=-1, gamma=0.9, temper=3.0):
        exit_layer = self.num_layers
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        img_type_ids = batch['img_type_ids']

        try:


            img_txt_encoder_outputs = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False,
                                          img_type_ids=img_type_ids)
            img_txt_sequence_output = img_txt_encoder_outputs[0]
            img_txt_logits, img_txt_pooled_output = self.output(img_txt_sequence_output, attn_masks)


            if self.training:

                img_encoder_outputs = self.uniter(None, position_ids,
                                                  img_feat, img_pos_feat,
                                                  attn_masks, gather_index,
                                                   output_all_encoded_layers=False,
                                                  img_type_ids=img_type_ids)

                txt_encoder_outputs = self.uniter(input_ids, position_ids,
                                                  None, img_pos_feat,
                                                  attn_masks, gather_index,
                                                  output_all_encoded_layers=False,
                                                  img_type_ids=img_type_ids)

                img_attention_mask = torch.Tensor([[1] * (img_feat.shape[1])] * img_feat.shape[0]).cuda()
                txt_attention_mask = torch.Tensor([[1] * (input_ids.shape[1])] * input_ids.shape[0]).cuda()

                img_sequence_output = img_encoder_outputs[0]
                img_logits, img_pooled_output = self.output(img_sequence_output, img_attention_mask)

                txt_sequence_output = txt_encoder_outputs[0]
                txt_logits , txt_pooled_output= self.output(txt_sequence_output, txt_attention_mask)

                outputs = (img_txt_logits,) + (img_logits,) + (txt_logits,) + (img_encoder_outputs[2:],) + (txt_encoder_outputs[2:],) + (img_txt_encoder_outputs[2:],)
            else:
                outputs = (img_txt_logits,) + (img_txt_encoder_outputs[2:],)

        except EElayerException as e:
            outputs = e.message   #  # EElogits, all_hidden_states, all_attentions,all_EElayer_exits
            exit_layer = e.exit_layer
            img_txt_logits = outputs[0]
            img_txt_pooled_output = outputs[1]
            img_txt_sequence_output = outputs[2]


        img_txt_eachlayer_logits_all = []
        img_txt_eachlayer_feature_all = []
        img_txt_eachlayer_hidden_feature_all = []

        img_eachlayer_logits_all = []
        img_eachlayer_feature_all = []
        img_eachlayer_hidden_feature_all = []

        txt_eachlayer_logits_all = []
        txt_eachlayer_feature_all = []
        txt_eachlayer_hidden_feature_all = []

        if not self.training:
            original_entropy = entropy(img_txt_logits)
            eachlayer_entropy = []


        for img_txt_EElayer_exit in outputs[-1]:
            img_txt_EElayer_logits = img_txt_EElayer_exit[0]
            img_txt_EElayer_features = img_txt_EElayer_exit[1]
            img_txt_hidden = img_txt_EElayer_exit[2]

            img_txt_eachlayer_logits_all.append(img_txt_EElayer_logits)
            img_txt_eachlayer_feature_all.append(img_txt_EElayer_features)
            img_txt_eachlayer_hidden_feature_all.append(img_txt_hidden)

            if not self.training:
                eachlayer_entropy.append(img_txt_EElayer_exit[2])

        if self.training:
            for img_EElayer_exit in outputs[-3]:
                img_EElayer_logits = img_EElayer_exit[0]
                img_EElayer_features = img_EElayer_exit[1]
                img_hidden = img_EElayer_exit[2]

                img_eachlayer_logits_all.append(img_EElayer_logits)
                img_eachlayer_feature_all.append(img_EElayer_features)
                img_eachlayer_hidden_feature_all.append(img_hidden)


            for txt_EElayer_exit in outputs[-2]:
                txt_EElayer_logits = txt_EElayer_exit[0]
                txt_EElayer_features = txt_EElayer_exit[1]
                txt_hidden = txt_EElayer_exit[2]

                txt_eachlayer_logits_all.append(txt_EElayer_logits)
                txt_eachlayer_feature_all.append(txt_EElayer_features)
                txt_eachlayer_hidden_feature_all.append(txt_hidden)


        if compute_loss:
            labels = batch['targets']

            loss_fct = FKDLoss(len(outputs[-1]), gamma, temper, self.num_labels)
            img_txt_soft_labels = img_txt_logits.detach()
            img_txt_logits_loss_kd = loss_fct(img_txt_logits, img_txt_pooled_output, img_txt_sequence_output, img_txt_eachlayer_logits_all, img_txt_eachlayer_feature_all, img_txt_eachlayer_hidden_feature_all, labels, img_txt_soft_labels)

            if self.training:
                img_soft_labels = img_logits.detach()
                txt_soft_labels = txt_logits.detach()

                img_logits_loss_kd = loss_fct(img_logits, img_pooled_output, img_sequence_output, img_eachlayer_logits_all, img_eachlayer_feature_all,  img_eachlayer_hidden_feature_all, labels, img_soft_labels)
                txt_logits_loss_kd = loss_fct(txt_logits, txt_pooled_output, txt_sequence_output, txt_eachlayer_logits_all, txt_eachlayer_feature_all, txt_eachlayer_hidden_feature_all, labels, txt_soft_labels)


                loss_kd = 0.15 * img_logits_loss_kd[0] + 0.15 * txt_logits_loss_kd[0]+ 0.7 * img_txt_logits_loss_kd[0]
            else:
                loss_kd = img_txt_logits_loss_kd

            outputs = (loss_kd,) + outputs

        if not self.training:
                outputs = outputs + ((original_entropy, eachlayer_entropy), exit_layer)
                if output_layer >= 0:
                    outputs = (outputs[0],) + \
                              (img_txt_eachlayer_logits_all[output_layer],) + \
                              outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions), (entropies), (exit_layer)