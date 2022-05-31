import copy
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Config, T5PreTrainedModel, T5Stack
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

###############################################################################
############################## util for T5 model ##############################
###############################################################################

class FCLayer(nn.Module):
    """ R-BERT: https://github.com/monologg/R-BERT """
    # both attention dropout and fc dropout is 0.1 on Roberta: https://arxiv.org/pdf/1907.11692.pdf
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU() # roberta and electra both uses gelu whereas BERT used tanh

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return self.linear(x)


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


###############################################################################
############################ Constructed T5 model #############################
###############################################################################

# https://github.com/monologg/EncT5/blob/master/enc_t5/modeling_enc_t5.py
class IBPoolerForSequenceClassification(T5PreTrainedModel):
    """ 
    Using [CLS token hidden states, Input Sequence Weighted Average, SEP token hidden states] for classification
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1, pooler='adaptive'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        pooler_class = AdaptivePooler if pooler == 'adaptive' else MeanPooler
        self.pooler = pooler_class(input_size=config.hidden_size)
        
        # FC Layers
        self.cls_fc_layer = FCLayer(
            self.config.hidden_size * 4, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )

        self.sep_fc_layer = FCLayer(
            self.config.hidden_size * 4, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )

        self.label_classifier = FCLayer(
            self.config.hidden_size* 3, 
            self.config.num_labels,
            self.config.dropout_rate,
            use_activation=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        last_token_index = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # get adaptive pooler for the given sequence
        weights, pooled_output = self.pooler(outputs[0], mask=attention_mask)

        # Global token's 4 hidden states concatenation and  projection result
        idx_seq = torch.arange(input_ids.size(0)).to(input_ids.device)
        
        cls_concat = torch.cat(tuple([outputs["hidden_states"][i][idx_seq, 0, :] for i in [-4, -3, -2, -1]]), dim=-1)
        cls_output = self.cls_fc_layer(cls_concat)
        
        sep_concat = torch.cat(tuple([outputs["hidden_states"][i][idx_seq, last_token_index] for i in [-4, -3, -2, -1]]), dim=-1)
        sep_output = self.sep_fc_layer(sep_concat)
        
        concat = torch.cat(
            [
              cls_output,
              pooled_output,
              sep_output
            ], dim=-1
        )
        
        logits = self.label_classifier(concat)

        loss = None
        outputs.hidden_states = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# https://github.com/monologg/EncT5/blob/master/enc_t5/modeling_enc_t5.py
class EncT5ForSequenceClassification(T5PreTrainedModel):
    """ 
    Baseline T5 encoder model for classification using cls token 
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# T5ForSequenceClassification: https://gist.github.com/avidale/364ebc397a6d3425b1eba8cf2ceda525
# T5 source: https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/t5/modeling_t5.py
class T5ForSequenceClassification(T5PreTrainedModel):
    """ 
    T5 model from input sequence(code1+code2+special tokens) pooling for classification 
    """
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

# https://github.com/monologg/EncT5/blob/master/enc_t5/modeling_enc_t5.py
class IBVStackT5ForSequenceClassification(T5PreTrainedModel):
    """ 
    Extracting hidden states of cls_token and sep_token for classification
    Improved Baseline method
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.cls_fc_layer = FCLayer(
            self.config.hidden_size* 4, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )

        self.sep_fc_layer = FCLayer(
            self.config.hidden_size* 4, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )

        self.label_classifier = FCLayer(
            self.config.hidden_size * 2, 
            self.config.num_labels,
            self.config.dropout_rate,
            use_activation=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        last_token_index = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Global token's 4 hidden states concatenation and  projection result
        idx_seq = torch.arange(input_ids.size(0)).to(input_ids.device)
        
        cls_concat = torch.cat(tuple([outputs["hidden_states"][i][idx_seq, 0, :] for i in [-4, -3, -2, -1]]), dim=-1)
        cls_output = self.cls_fc_layer(cls_concat)
        
        sep_concat = torch.cat(tuple([outputs["hidden_states"][i][idx_seq, last_token_index] for i in [-4, -3, -2, -1]]), dim=-1)
        sep_output = self.sep_fc_layer(sep_concat)
        
        concat = torch.cat(
            [
              cls_output,
              sep_output
            ], dim=-1
        )
        
        logits = self.label_classifier(concat)

        loss = None
        outputs.hidden_states = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# https://github.com/monologg/EncT5/blob/master/enc_t5/modeling_enc_t5.py
class T5ConcatModel(T5PreTrainedModel):
    """ NOT WOKRING: Same structure as RBART """
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1, pooler='adaptive'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        pooler_class = AdaptivePooler if pooler == 'adaptive' else MeanPooler
        self.pooler = pooler_class(input_size=config.hidden_size)
        self.token_fc_layer = FCLayer(
            self.config.hidden_size, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )

        self.entity_fc_layer = FCLayer(
            self.config.hidden_size, 
            self.config.hidden_size, 
            self.config.dropout_rate,
            use_activation=True,
        )
        self.label_classifier = FCLayer(
            self.config.hidden_size * 3,
            self.config.num_labels,
            self.config.dropout_rate,
            use_activation=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self, 
        input_ids, 
        attention_mask,
        hypothesis_mask,
        premise_mask, 
        last_token_index,
        labels=None
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        # Global average on sentences
        sequence_output = outputs["last_hidden_state"]
        sentence_h = self.entity_average(sequence_output, attention_mask)
        # sentence_h = self.entity_fc_layer(sentence_h)
        # print(sentence_h)
        
        premise_sentence_h = self.entity_average(sequence_output, premise_mask) # token in between subject entities -> 
        # premise_sentence_h = self.entity_fc_layer(premise_sentence_h) # subject entity's fully connected layer | yellow on diagram
        # print(premise_sentence_h)
        
        # Average on hypothesis sentence
        hypothesis_sentence_h = self.entity_average(sequence_output, hypothesis_mask) # token in between object entities
        # hypothesis_sentence_h = self.entity_fc_layer(hypothesis_sentence_h) # object entity's fully connected layer | red on diagram
        # print(hypothesis_sentence_h)
        
        # Concat: global token, global average, premise token, premise average, hypothesis token, hypothesis average
        concat = torch.cat(
            [
                # cls_output, 
                sentence_h, 
                premise_sentence_h, 
                hypothesis_sentence_h
            ], dim=-1
        )
        
        # yield logit from label classifier
        logits = self.label_classifier(concat)
        
        loss = None
        outputs.hidden_states = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
