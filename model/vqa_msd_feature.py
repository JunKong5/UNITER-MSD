"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from .model_msd_feature import EElayerException
from .layer import GELU
from .model_msd_feature import UniterPreTrainedModel, UniterModel










class FKDLoss(nn.Module):
    def __init__(self, nBlocks, gamma, T, num_labels, hidden_size):
        super(FKDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss().cuda()
        self.VID_Loss = VIDLoss(hidden_size,hidden_size)
        self.ce_loss = F.binary_cross_entropy_with_logits
        self.mse_loss = nn.MSELoss().cuda()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.nBlocks = nBlocks
        self.gamma = gamma
        self.T = T
        self.num_labels = num_labels

    def forward(self, outputs, last_hidden_feature, highway_outputs, eachlayer_hiden_feature_all,targets, soft_targets):
        if self.num_labels == 1:
            last_loss = self.mse_loss(outputs, targets)
        else:

            last_loss = self.ce_loss(outputs, targets)

        T = self.T
        multi_losses = []
        distill_losses = []
        kls = []
        hidden_feature_losses = []
        for i in range(self.nBlocks - 1):
            if self.num_labels == 1:
                _mse = (1. - self.gamma) * self.mse_loss(highway_outputs[i], targets)
                _kld = self.kld_loss(self.log_softmax(highway_outputs[i] / T),
                                     self.softmax(soft_targets / T)) * self.gamma * T * T
                multi_losses.append(_mse)
                distill_losses.append(_kld)
            else:

                _ce = (1. - self.gamma) * self.ce_loss(highway_outputs[i], targets)

                _kld = self.kld_loss(self.log_softmax(highway_outputs[i] / T),
                                     self.softmax(soft_targets / T)) * self.gamma * T * T
                _klds = self.kld_loss(self.log_softmax(highway_outputs[i] / T),
                                     self.softmax(soft_targets / T))

                kls.append(_klds)
                hidden_feature = self.mse_loss(last_hidden_feature,eachlayer_hiden_feature_all[i])*1e-6

                multi_losses.append(_ce)
                distill_losses.append(_kld)
                hidden_feature_losses.append(hidden_feature)
        kls = sum(kls)/12
        m_loss = sum(multi_losses)
        d_loss = sum(distill_losses)
        hf_loss = sum(hidden_feature_losses)
        l_loss = last_loss


        loss = l_loss + d_loss + m_loss  + hf_loss


        return (loss, l_loss, d_loss, m_loss,hf_loss,kls)

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim,num_answer)
        self.num_layers = config.num_hidden_layers
        self.num_labels = num_answer
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, output_layer=-1, gamma=0.9, temper=3.0):
        exit_layer = self.num_layers
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']

        try:
            img_txt_encoder_outputs = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attn_masks, gather_index,
                                          output_all_encoded_layers=False)
            img_txt_hidden_feature = img_txt_encoder_outputs[0]

            img_txt_pooled_output = img_txt_encoder_outputs[1]
            # pooled_output = self.dropout(pooled_output)
            img_txt_logits = self.vqa_output(img_txt_pooled_output)


            if self.training:

                img_encoder_outputs = self.uniter(None, position_ids,
                                                  img_feat, img_pos_feat,
                                                  attn_masks, gather_index,
                                                   output_all_encoded_layers=False)

                txt_encoder_outputs = self.uniter(input_ids, position_ids,
                                                  None, img_pos_feat,
                                                  attn_masks, gather_index,
                                                  output_all_encoded_layers=False)

                img_hidden_feature = img_encoder_outputs[0]
                img_pooled_output = img_encoder_outputs[1]
                img_logits = self.vqa_output(img_pooled_output)

                txt_hidden_feature = txt_encoder_outputs[0]
                txt_pooled_output = txt_encoder_outputs[1]
                txt_logits = self.vqa_output(txt_pooled_output)

                outputs = (img_txt_logits,) + (img_logits,) + (txt_logits,) +(img_encoder_outputs[2:],)+(txt_encoder_outputs[2:],) + (img_txt_encoder_outputs[2:],)
            else:
                outputs = (img_txt_logits,) + (img_txt_encoder_outputs[2:],)



        except EElayerException as e:
            outputs = e.message   #  # EElogits, EEfeature, all_hidden_states, all_attentions, all_EElayer_exits
            exit_layer = e.exit_layer
            img_txt_logits = outputs[0]
            img_txt_pooled_output = outputs[1]
            img_txt_hidden_feature = outputs[2]

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


        for img_txt_EElayer_exit in outputs[-1]:                #(all_EElayer_exits,)
            img_txt_EElayer_logits = img_txt_EElayer_exit[0]
            img_txt_EElayer_features = img_txt_EElayer_exit[1]
            img_txt_hidden = img_txt_EElayer_exit[2]

            img_txt_eachlayer_logits_all.append(img_txt_EElayer_logits)
            img_txt_eachlayer_feature_all.append(img_txt_EElayer_features)
            img_txt_eachlayer_hidden_feature_all.append(img_txt_hidden)
            if not self.training:
                eachlayer_entropy.append(img_txt_EElayer_exit[-1])

        if self.training:
            for img_EElayer_exit in outputs[3]:
                img_EElayer_logits = img_EElayer_exit[0]
                img_EElayer_features = img_EElayer_exit[1]
                img_hidden = img_EElayer_exit[2]

                # print("img_hidden",img_hidden.shape)

                img_eachlayer_hidden_feature_all.append(img_hidden)
                img_eachlayer_logits_all.append(img_EElayer_logits)
                img_eachlayer_feature_all.append(img_EElayer_features)


            for txt_EElayer_exit in outputs[4]:
                txt_EElayer_logits = txt_EElayer_exit[0]
                txt_EElayer_features = txt_EElayer_exit[1]
                txt_hidden = txt_EElayer_exit[2]

                # print("txt_hidden", txt_hidden.shape)

                txt_eachlayer_hidden_feature_all.append(txt_hidden)
                txt_eachlayer_logits_all.append(txt_EElayer_logits)
                txt_eachlayer_feature_all.append(txt_EElayer_features)


        if compute_loss:
            labels = batch['targets']

            loss_fct = FKDLoss(len(outputs[-1]), gamma, temper, self.num_labels,self.hidden_size)
            img_txt_soft_labels = img_txt_logits.detach()
            img_txt_logits_loss_kd = loss_fct(img_txt_logits, img_txt_pooled_output,img_txt_hidden_feature, img_txt_eachlayer_logits_all, img_txt_eachlayer_feature_all, img_txt_eachlayer_hidden_feature_all, labels, img_txt_soft_labels)
            if self.training:
                img_soft_labels = img_logits.detach()
                txt_soft_labels = txt_logits.detach()

                img_logits_loss_kd = loss_fct(img_logits, img_pooled_output, img_hidden_feature, img_eachlayer_logits_all, img_eachlayer_feature_all, img_eachlayer_hidden_feature_all, labels, img_soft_labels)
                txt_logits_loss_kd = loss_fct(txt_logits, txt_pooled_output, txt_hidden_feature, txt_eachlayer_logits_all, txt_eachlayer_feature_all, txt_eachlayer_hidden_feature_all, labels, txt_soft_labels)


                loss_kd = 0.2 * img_logits_loss_kd[0] + 0.2 * txt_logits_loss_kd[0]+ 0.6 * img_txt_logits_loss_kd[0]
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
