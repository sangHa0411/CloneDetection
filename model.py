import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from transformers import RobertaPreTrainedModel, RobertaModel
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
            nn.ReLU()
        )
        self.classifier = nn.Linear(config.hidden_size*4, config.num_labels)

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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        cls_flag = input_ids == self.config.tokenizer_cls_token_id # cls token
        sep_flag = input_ids == self.config.tokenizer_sep_token_id # sep token 

        sep_token_states = hidden_states[cls_flag + sep_flag] # (batch_size * 4, hidden_size)
        sep_token_states = sep_token_states.view(batch_size, -1, self.config.hidden_size) # (batch_size, 4, hidden_size)
        sep_hidden_states = self.net(sep_token_states) # (batch_size, 4, hidden_size)

        pooled_output = sep_hidden_states.view(batch_size, -1) # (batch_size, hidden_size * 4)
        logits = self.classifier(pooled_output) # (batch_size, 2)

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
    