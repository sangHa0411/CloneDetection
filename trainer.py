
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

class Trainer(Trainer) :
    def get_kl_loss(self, loss_fn, logits_1, logits_2, alpha=1) :
        loss_kl_1 = loss_fn(F.log_softmax(logits_1, dim=-1), F.softmax(logits_2, dim=-1))
        loss_kl_2 = loss_fn(F.log_softmax(logits_2, dim=-1), F.softmax(logits_1, dim=-1))
        return alpha * (loss_kl_1 + loss_kl_2) / 2

    def compute_eval_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=inputs['labels']
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():               
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_eval_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)

    def compute_loss(self, model, inputs):
        num_labels = model.config.num_labels

        labels = inputs.pop('labels')

        outputs1 = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits1 = outputs1.logits

        outputs2 = model(input_ids=inputs['input_ids2'], attention_mask=inputs['attention_mask2'])
        logits2 = outputs2.logits

        # Crossentropy Loss    
        loss_fct_1 = nn.CrossEntropyLoss()
        loss_nll = loss_fct_1(logits1.view(-1, num_labels), labels.view(-1)) + loss_fct_1(logits2.view(-1, num_labels), labels.view(-1))

        # KL-Divergence Loss
        loss_fct_2 = nn.KLDivLoss(reduction='batchmean')
        loss_kl = self.get_kl_loss(loss_fct_2, logits1, logits2)
        return loss_nll + loss_kl
