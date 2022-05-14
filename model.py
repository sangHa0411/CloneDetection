import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import copy
from typing import Optional, Union, Tuple
from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    T5PreTrainedModel,
    PLBartPreTrainedModel,
    PLBartModel,
)
from transformers.models.t5.modeling_t5 import T5Stack, T5EncoderModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput


class RobertaClassificationHead(nn.Module):
    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaRBERT(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_size = len(input_ids)
        hidden_states = outputs[0]

        # CLS code1 SEP SEP code2 SEP
        cls_flag = input_ids == self.config.tokenizer_cls_token_id  # cls token
        sep_flag = input_ids == self.config.tokenizer_sep_token_id  # sep token

        sep_token_states = hidden_states[
            cls_flag + sep_flag
        ]  # (batch_size * 4, hidden_size)
        sep_token_states = sep_token_states.view(
            batch_size, -1, self.config.hidden_size
        )  # (batch_size, 4, hidden_size)
        sep_hidden_states = self.net(sep_token_states)  # (batch_size, 4, hidden_size)

        pooled_output = sep_hidden_states.view(
            batch_size, -1
        )  # (batch_size, hidden_size * 4)
        logits = self.classifier(pooled_output)  # (batch_size, 2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# BartForSequenceclassification: https://github.com/huggingface/transformers/blob/a59eb349c5616c1b48ae9225028fb41ec1feb6aa/src/transformers/models/bart/modeling_bart.py#L1437
# T5ForSequenceClassification: https://gist.github.com/avidale/364ebc397a6d3425b1eba8cf2ceda525
# T5 source: https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/t5/modeling_t5.py


def mean_pooling(inputs, mask):
    token_embeddings = inputs
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MeanPooler(nn.Module):
    """Calcualte simple average of the inputs"""

    def __init__(self, input_size=None):
        super().__init__()

    def forward(self, inputs, mask=None):
        if mask is None:
            pooled_output = inputs.mean(dim=1)
        else:
            pooled_output = mean_pooling(inputs, mask)
        return None, pooled_output


class AdaptivePooler(nn.Module):
    """Calcualte weighted average of the inputs with learnable weights"""

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.w = nn.Linear(self.input_size, 1, bias=True)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, emb_dim = inputs.shape
        scores = torch.squeeze(self.w(inputs), dim=-1)
        weights = nn.functional.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdims=True)
        outputs = (inputs.permute(2, 0, 1) * weights).sum(-1).T
        return weights, outputs


class T5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config, pooler="adaptive"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = T5EncoderModel.from_pretrained(
            config.model_name_or_path, config=encoder_config
        )

        pooler_class = AdaptivePooler if pooler == "adaptive" else MeanPooler
        self.pooler = pooler_class(input_size=config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        weights, pooled_output = self.pooler(outputs[0], mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# https://github.com/wzhouad/RE_improved_baseline/blob/main/model.py
# https://github.com/huggingface/transformers/blob/4975002df50c472cbb6f8ac3580e475f570606ab/src/transformers/models/plbart/modeling_plbart.py#L1110-L1219
# https://dacon.io/competitions/official/235875/codeshare/4589?page=1&dtype=recent
class BartEncoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BartEncoderConcatModel(
    PLBartPreTrainedModel
):  # PLBartModel하고 self.encoder해도 될 듯?
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.bart_model = PLBartModel.from_pretrained(
            config.model_name_or_path, config=config
        )
        self.encoder = self.bart_model.encoder
        self.tokenizer = tokenizer
        self.classifier = BartEncoderClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        first_token_index=None,
        second_token_index=None,
        labels=None,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        last_hidden_state = outputs[
            "last_hidden_state"
        ]  # or outputs["hidden_state"][-1]
        idx_seq = torch.arange(input_ids.size(0)).to(input_ids.device)
        first_token_vec = last_hidden_state[idx_seq, first_token_index]
        second_token_vec = last_hidden_state[idx_seq, second_token_index]
        concat_vec = torch.cat([first_token_vec, second_token_vec], dim=-1)

        logits = self.classifier(concat_vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(logits)
            # print(labels)
            labels = labels.squeeze(-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
