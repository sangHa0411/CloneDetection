from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_ids2 = [feature["input_ids2"] for feature in features] if "input_ids2" in features[0].keys() else None
        if input_ids2 is not None:
            max_input_ids2_length = max(len(l) for l in input_ids2)
            for feature in features:
                feature["input_ids2"] = feature["input_ids2"] + [self.tokenizer.pad_token_id] * (max_input_ids2_length - len(feature["input_ids2"]))
                feature["attention_mask2"] = feature["attention_mask2"] + [0] * (max_input_ids2_length - len(feature["input_ids2"]))
                
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch