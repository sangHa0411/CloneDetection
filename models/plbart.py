import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PLBartPreTrainedModel,
    PLBartModel,
)

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
