import random
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding


def add_entity_mask(
    input_id_len,
    first_token_start_index,
    first_token_end_index,
    second_token_start_index,
    second_token_end_index,
):
    """add entity token to input_ids"""
    # print("tokenized input ids: \n",item['input_ids'])

    # initialize entity masks
    premise_mask = np.zeros((input_id_len,), dtype=int)
    hypothesis_mask = np.zeros((input_id_len,), dtype=int)

    premise_mask[first_token_start_index:first_token_end_index] = 1

    hypothesis_mask[second_token_start_index:second_token_end_index] = 1

    return premise_mask.astype(int), hypothesis_mask.astype(int)


class Encoder:
    def __init__(self, tokenizer, model_category, max_input_length: int):
        self.tokenizer = tokenizer
        self.model_category = model_category
        self.max_input_length = max_input_length

    def __call__(self, examples):

        if self.model_category == "codebert":
            code1_inputs = self.tokenizer(
                examples["code1"],
                examples["code2"],
                max_length=self.max_input_length,
                return_token_type_ids=False,
                truncation=True,
            )

            code2_inputs = self.tokenizer(
                examples["code2"],
                examples["code1"],
                max_length=self.max_input_length,
                return_token_type_ids=False,
                truncation=True,
            )

            model_inputs = BatchEncoding(
                {
                    "input_ids": code1_inputs["input_ids"],
                    "attention_mask": code1_inputs["attention_mask"],
                    "input_ids2": code2_inputs["input_ids"],
                    "attention_mask2": code2_inputs["attention_mask"],
                }
            )
        elif self.model_category == "t5":
            batch_size = len(examples["code1"])
            max_input_length = int(self.max_input_length / 2)

            code1_inputs = self.tokenizer(
                examples["code1"],
                max_length=max_input_length,
                return_token_type_ids=False,
                truncation=True,
            )

            code2_inputs = self.tokenizer(
                examples["code2"],
                max_length=max_input_length,
                return_token_type_ids=False,
                truncation=True,
            )

            input_ids1 = []
            input_ids2 = []
            attention_mask1 = []
            attention_mask2 = []

            for i in range(batch_size):
                input_ids1.append(
                    [self.tokenizer.cls_token_id]
                    + code1_inputs["input_ids"][i]
                    + [self.tokenizer.sep_token_id]
                    + code2_inputs["input_ids"][i]
                )
                input_ids2.append(
                    [self.tokenizer.cls_token_id]
                    + code2_inputs["input_ids"][i]
                    + [self.tokenizer.sep_token_id]
                    + code1_inputs["input_ids"][i]
                )

                attention_mask1.append(
                    [1]
                    + code1_inputs["attention_mask"][i]
                    + [1]
                    + code2_inputs["attention_mask"][i]
                )
                attention_mask2.append(
                    [1]
                    + code2_inputs["attention_mask"][i]
                    + [1]
                    + code1_inputs["attention_mask"][i]
                )

            model_inputs = BatchEncoding(
                {
                    "input_ids": input_ids1,
                    "attention_mask": attention_mask1,
                    "input_ids2": input_ids2,
                    "attention_mask2": attention_mask2,
                }
            )
        elif self.model_category == "plbart":
            # define special token if using plbart
            python_token = self.tokenizer(
                "python", max_length=1, add_special_tokens=False, truncation=True
            ).input_ids

            # define list of items to be returned
            list_input_ids = []
            list_attention_mask = []
            list_hypothesis_mask = []
            list_premise_mask = []
            list_last_token_index = []
            batch_size = len(examples["code1"])
            for i in range(batch_size):
                # randomly switching code1 and code2
                k = random.randint(0, 1)  # decide on k once
                if k == 0:
                    code1 = examples["code1"][i]
                    code2 = examples["code2"][i]
                elif k == 1:
                    code2 = examples["code1"][i]
                    code1 = examples["code2"][i]

                code1_input_ids = self.tokenizer(
                    code1,
                    max_length=self.max_input_length,
                    add_special_tokens=False,
                    truncation=True,
                ).input_ids
                code2_input_ids = self.tokenizer(
                    code2,
                    max_length=self.max_input_length,
                    add_special_tokens=False,
                    truncation=True,
                ).input_ids
                len_input_ids = (
                    len(code1_input_ids) + len(code2_input_ids) + 2 + 2 + 1
                )  # python </s></s> python </s>

                # normalize length for overflowing codes by truncation
                if len_input_ids >= self.max_input_length:
                    # get tokenized length of each codes
                    code1_length_bart, code2_length_bart = len(code1_input_ids) / (
                        self.max_input_length
                    ), len(code2_input_ids) / (self.max_input_length)
                    # softmax to make normalized sum to be come (self.max_input_length)
                    codes_length = [code1_length_bart, code2_length_bart]
                    codes_length = np.exp(codes_length) / np.sum(np.exp(codes_length))
                    normalized_code_length = codes_length * (self.max_input_length)
                    normalized_code_length = np.round(normalized_code_length).astype(
                        int
                    )
                    code1_input_ids = self.tokenizer(
                        code1,
                        max_length=normalized_code_length[0] - 2,
                        add_special_tokens=False,
                        truncation=True,
                    ).input_ids

                    code2_input_ids = self.tokenizer(
                        code2,
                        max_length=normalized_code_length[1] - 2,
                        add_special_tokens=False,
                        truncation=True,
                    ).input_ids

                else:
                    pass

                # construct input sequence
                input_ids = (
                    code1_input_ids
                    + python_token
                    + [self.tokenizer.sep_token_id]
                    + [self.tokenizer.sep_token_id]
                    + code2_input_ids
                    + python_token
                )

                # exception handling when constructed sequence overflows
                if len(input_ids) >= self.max_input_length:
                    input_ids = input_ids[: self.max_input_length]
                    input_ids[-2] = python_token[0]
                    input_ids[-1] = self.tokenizer.sep_token_id
                elif len(input_ids) < self.max_input_length:
                    input_ids += [self.tokenizer.sep_token_id]

                # input token indexing for Improved Baseline format
                first_token_start_index = 0
                first_python_index = first_token_start_index + len(code1_input_ids)

                first_sep_token_index = first_python_index + 1
                second_sep_token_index = first_sep_token_index + 1
                second_python_index = second_sep_token_index + len(code2_input_ids) + 1

                if len(input_ids) >= self.max_input_length:
                    third_sep_token_index = self.max_input_length - 1
                    second_python_index = third_sep_token_index - 1
                elif len(input_ids) < self.max_input_length:
                    third_sep_token_index = second_python_index + 1

                premise_mask, hypothesis_mask = add_entity_mask(
                    len(input_ids),
                    first_token_start_index,
                    first_sep_token_index,
                    second_sep_token_index + 1,
                    third_sep_token_index,
                )

                attention_mask = np.array(
                    [1] * len(input_ids)
                    + [0] * max(0, self.max_input_length - len(input_ids))
                )
                # print(premise_mask)
                padding = [0] * max(0, self.max_input_length - len(input_ids))
                if len(input_ids) < self.max_input_length:
                    input_ids = np.concatenate((input_ids, padding), axis=None)
                premise_mask = np.concatenate(
                    (premise_mask.astype(int), padding), axis=None
                )
                premise_mask = premise_mask.astype(int)
                hypothesis_mask = np.concatenate(
                    (hypothesis_mask.astype(int), padding), axis=None
                )
                hypothesis_mask = hypothesis_mask.astype(int)

                list_input_ids.append(input_ids)
                list_attention_mask.append(attention_mask)
                list_hypothesis_mask.append(hypothesis_mask)
                list_premise_mask.append(premise_mask)
                # list_labels.append
                list_last_token_index.append(third_sep_token_index)

            # Wrap as batch format
            model_inputs = BatchEncoding(
                {
                    "input_ids": list_input_ids,
                    "attention_mask": list_attention_mask,
                    "hypothesis_mask": list_hypothesis_mask,
                    "premise_mask": list_premise_mask,
                    "last_token_index": list_last_token_index,
                }
            )

        if "similar" in examples:
            model_inputs["labels"] = examples["similar"]
        return model_inputs
