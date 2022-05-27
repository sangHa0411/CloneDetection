import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.models.plbart.configuration_plbart import PLBartConfig
from transformers.models.plbart.modeling_plbart import PLBartEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    PLBartPreTrainedModel,
    PLBartModel,
)


class FCLayer(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT"""

    # both attention dropout and fc dropout is 0.1 on Roberta: https://arxiv.org/pdf/1907.11692.pdf
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = (
            nn.GELU()
        )  # roberta and electra both uses gelu whereas BERT used tanh

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return self.linear(x)


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


class PLBartClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states[:, 0, :])
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# BartForSequenceclassification: https://github.com/huggingface/transformers/blob/a59eb349c5616c1b48ae9225028fb41ec1feb6aa/src/transformers/models/bart/modeling_bart.py#L1437
# https://github.com/wzhouad/RE_improved_baseline/blob/main/model.py
# https://github.com/huggingface/transformers/blob/4975002df50c472cbb6f8ac3580e475f570606ab/src/transformers/models/plbart/modeling_plbart.py#L1110-L1219
# https://dacon.io/competitions/official/235875/codeshare/4589?page=1&dtype=recent
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


class PLEncoderForCloneDetection(PLBartPreTrainedModel):
    def __init__(self, config: PLBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PLBartEncoder(config, self.shared)

        self.classification_head = PLBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
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

        last_hidden_states = outputs[0]  # last hidden state
        logits = self.classification_head(last_hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# https://github.com/monologg/R-BERT/blob/master/model.py
# https://dacon.io/competitions/official/235875/codeshare/4589?page=1&dtype=recent
class RBartConcatModel(PLBartModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.entity_fc_layer = FCLayer(
            self.config.hidden_size, self.config.hidden_size, self.config.dropout_rate
        )
        # self.proj_fc_layer = FCLayer(self.config.hidden_size * 4, self.config.hidden_size, self.config.dropout_rate)
        self.label_classifier = FCLayer(
            self.config.hidden_size * 3,
            self.config.num_labels,
            self.config.dropout_rate,
            use_activation=True,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        hypothesis_mask,
        premise_mask,
        last_token_index,
        labels,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # Global token's 4 hidden states concatenation and  projection result
        # idx_seq = torch.arange(input_ids.size(0)).to(input_ids.device)
        # print(idx_seq)
        # cls_concat = torch.cat(tuple([outputs["hidden_states"][i][:, last_token_index, :] for i in [-4, -3, -2, -1]]), dim=-1)
        # cls_output = self.proj_fc_layer(cls_concat).squeeze(0)
        # cls_output = outputs["last_hidden_state"][:, last_token_index, :]
        # print(cls_output)

        # Global average on sentences
        sequence_output = outputs["last_hidden_state"]
        sentence_h = self.entity_average(sequence_output, attention_mask)
        sentence_h = self.entity_fc_layer(sentence_h)
        # print(sentence_h)

        premise_sentence_h = self.entity_average(
            sequence_output, premise_mask
        )  # token in between subject entities ->
        premise_sentence_h = self.entity_fc_layer(
            premise_sentence_h
        )  # subject entity's fully connected layer | yellow on diagram
        # print(premise_sentence_h)

        # Average on hypothesis sentence
        hypothesis_sentence_h = self.entity_average(
            sequence_output, hypothesis_mask
        )  # token in between object entities
        hypothesis_sentence_h = self.entity_fc_layer(
            hypothesis_sentence_h
        )  # object entity's fully connected layer | red on diagram
        # print(hypothesis_sentence_h)

        # Concat: global token, global average, premise token, premise average, hypothesis token, hypothesis average
        concat = torch.cat(
            [
                # cls_output,
                sentence_h,
                premise_sentence_h,
                hypothesis_sentence_h,
            ],
            dim=-1,
        )

        # yield logit from label classifier
        logits = self.label_classifier(concat)
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

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
