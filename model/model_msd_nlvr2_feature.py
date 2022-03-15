"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
from .layer import GELU
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from .layer import BertLayer, BertPooler
from torch.nn import functional as F


logger = logging.getLogger(__name__)

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

class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

from .attention import MultiheadAttention
class BertEElayer (nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final BertLayer in BertEncoder
    to
    cross-entropy computation in BertForSequenceClassification
    """
    def __init__(self, config, num_answer ):
        super(BertEElayer, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_answer )

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

        self.nlvr2_output = nn.Linear(2 * config.hidden_size, 2)

    def forward(self, encoder_outputs,attn_masks):
        # separate left image and right image
        encoder_outputs = encoder_outputs[0]
        bs, tl, d = encoder_outputs.size()
        left_out, right_out = encoder_outputs.contiguous().view(
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

        return answer_scores , torch.cat([left_out, right_out], dim=-1),encoder_outputs



def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A

class EElayerException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!

class UniterEncoder(nn.Module):
    def __init__(self, config, num_answer):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.EElayer = nn.ModuleList([BertEElayer(config, num_answer ) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]


    def set_early_exit_entropy(self, x):
        print("set_early_exit_entropy",x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x





    def forward(self, input_, attention_mask,attention_mask1,
                output_all_encoded_layers=True):
        all_EElayer_exits = ()
        all_hidden_states = ()
        hidden_states = input_
        for i,layer_module in enumerate( self.layer):

            if output_all_encoded_layers:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states, attention_mask)
            current_outputs = (hidden_states,)

            if output_all_encoded_layers:
                current_outputs = current_outputs + (all_hidden_states,)

            EElayer_exit = self.EElayer[i](current_outputs,attention_mask1)

            if not self.training:
                EElayer_logits = EElayer_exit[0]
                EElayer_feature = EElayer_exit[1]
                hidden_feature = EElayer_exit[2]

                EElayer_entropy = entropy(EElayer_logits)
                EElayer_exit = EElayer_exit + (EElayer_entropy,)  # logits, pooled_output, entropy
                all_EElayer_exits = all_EElayer_exits + (EElayer_exit,)



                if EElayer_entropy < self.early_exit_entropy[i]:

                    new_output = (EElayer_logits,) + (EElayer_feature,) +(hidden_feature,)+ current_outputs[1:] + (all_EElayer_exits ,) # EElogits, all_hidden_states, all_attentions, all_EElayer_exits
                    raise EElayerException(new_output, i+1)

            else:
                all_EElayer_exits  = all_EElayer_exits  + (EElayer_exit,)        # numlayer * (logits  [bs,clasess], pooled_output [ba,hiden],entropy)


        outputs = (hidden_states,)

        outputs = outputs + (all_EElayer_exits,)

        return outputs  # last-layer hidden state, (all hidden states), all EElayer exits


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim,num_answer):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config, num_answer)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)
        self.num_answer =num_answer


    def init_early_exit_pooler(self):
        self.encoder.init_early_exit_pooler(self.pooler)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None: img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat, img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,img_feat, img_pos_feat,gather_index, img_masks=None, txt_type_ids=None, img_type_ids=None):

        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)

        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),dim=1, index=gather_index)
        return embedding_output

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):

        # compute self-attention mask

        if input_ids is None:
            attention_mask = torch.Tensor([[1] * (img_feat.shape[1])]*img_feat.shape[0])

        elif img_feat is None:
            attention_mask = torch.Tensor([[1] * (input_ids.shape[1])]*input_ids.shape[0])
        else:
            pass

        attention_mask = attention_mask.cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(input_ids, position_ids, img_feat, img_pos_feat, gather_index, img_masks, txt_type_ids, img_type_ids)

        encoder_outputs  = self.encoder(
            embedding_output, extended_attention_mask,attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1]
        return outputs
