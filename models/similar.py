import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import ModelOutput

@dataclass
class SimilarityOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states1: Optional[Tuple[torch.FloatTensor]] = None
    attentions1: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states2: Optional[Tuple[torch.FloatTensor]] = None
    attentions2: Optional[Tuple[torch.FloatTensor]] = None

class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features1, features2):
        x = features1[:, 0, :]  + features2[:, 0, :]
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

        self.reoberta_model1 = RobertaModel(config=config, add_pooling_layer=False)
        self.reoberta_model2 = RobertaModel(config=config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

    def load_weight(self, model_checkpoint) :
        self.reoberta_model1 = RobertaModel.from_pretrained(model_checkpoint, config=self.config, add_pooling_layer=False)
        self.reoberta_model2 = RobertaModel.from_pretrained(model_checkpoint, config=self.config, add_pooling_layer=False)

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
    ) -> Union[Tuple, SimilarityOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.reoberta_model1(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs2 = self.reoberta_model2(
            input_ids2,
            attention_mask=attention_mask2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logits = self.classifier(outputs1[0], outputs2[0])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs1[2:] + outputs2[2:]
            return ((loss,) + output) if loss is not None else output

        return SimilarityOutput(
            loss=loss,
            logits=logits,
            hidden_states1=outputs1.hidden_states,
            attentions1=outputs1.attentions,
            hidden_states2=outputs2.hidden_states,
            attentions2=outputs2.attentions,
        )