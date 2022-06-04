import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT"""

    # both attention dropout and fc dropout is 0.1 on Roberta: https://arxiv.org/pdf/1907.11692.pdf
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()  # roberta and electra both uses gelu whereas BERT used tanh

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
        self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float,
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


def mean_pooling(inputs, mask):
    token_embeddings = inputs
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MeanPooler(nn.Module):
    """ Calcualte simple average of the inputs """

    def __init__(self, input_size=None):
        super().__init__()

    def forward(self, inputs, mask=None):
        if mask is None:
            pooled_output = inputs.mean(dim=1)
        else:
            pooled_output = mean_pooling(inputs, mask)
        return None, pooled_output


class AdaptivePooler(nn.Module):
    """ Calcualte weighted average of the inputs with learnable weights """

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
