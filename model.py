from msilib import sequence
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import ModelOutput

@dataclass
class SimilarityOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states1: Optional[Tuple[torch.FloatTensor]] = None
    attentions1: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states1: Optional[Tuple[torch.FloatTensor]] = None
    attentions1: Optional[Tuple[torch.FloatTensor]] = None

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSimilarityClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta_code1 = RobertaModel(config, add_pooling_layer=False)
        self.roberta_code2 = RobertaModel(config, add_pooling_layer=False)

        # self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids2: Optional[torch.LongTensor] = None,
        attention_mask2: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SimilarityOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.roberta_code1(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs2 = self.roberta_code2(
            input_ids2,
            attention_mask=attention_mask2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        code1_sequence_output = outputs1[0].unsqueeze(1) # (batch_size, 1, seq_size)
        code2_sequence_output = outputs2[0].unsqueeze(-1) # (batch_size, seq_size, -1)

        code_similarity_output = torch.matmul(code1_sequence_output, code2_sequence_output)
        logits = code_similarity_output.squeeze() # (batch_size, )

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs1[2:] + outputs2[2:]
            return ((loss,) + output) if loss is not None else output

        return SimilarityOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states1=outputs1.hidden_states,
            attentions1=outputs1.attentions,
            hidden_states2=outputs2.hidden_states,
            attentions2=outputs2.attentions,
        )