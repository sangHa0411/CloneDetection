from transformers.tokenization_utils_base import BatchEncoding

class Encoder :
    def __init__(self, tokenizer, model_category, max_input_length: int) :
        self.tokenizer = tokenizer
        self.model_category = model_category
        self.max_input_length = max_input_length
    
    def __call__(self, examples):

        if self.model_category == 'codebert' :
            code1_inputs = self.tokenizer(examples['code1'], 
                examples['code2'],
                max_length=self.max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            code2_inputs = self.tokenizer(examples['code2'], 
                examples['code1'],
                max_length=self.max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            model_inputs = BatchEncoding({'input_ids' : code1_inputs['input_ids'],
                'attention_mask' : code1_inputs['attention_mask'],
                'input_ids2' : code2_inputs['input_ids'],
                'attention_mask2' : code2_inputs['attention_mask']
                }
            )
        else : # model_category == 'plbart' or 't5'
            batch_size = len(examples['code1'])
            max_input_length = int(self.max_input_length / 2)

            code1_inputs = self.tokenizer(examples['code1'], 
                max_length=max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            code2_inputs = self.tokenizer(examples['code2'], 
                max_length=max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            input_ids1 = []
            input_ids2 = []
            attention_mask1 = []
            attention_mask2 = []

            for i in range(batch_size) :
                input_ids1.append([self.tokenizer.cls_token_id] + code1_inputs['input_ids'][i] + [self.tokenizer.sep_token_id] + code2_inputs['input_ids'][i])
                input_ids2.append([self.tokenizer.cls_token_id] + code2_inputs['input_ids'][i] + [self.tokenizer.sep_token_id] + code1_inputs['input_ids'][i])

                attention_mask1.append([1] +  code1_inputs['attention_mask'][i] + [1] + code2_inputs['attention_mask'][i])
                attention_mask2.append([1] +  code2_inputs['attention_mask'][i] + [1] + code1_inputs['attention_mask'][i])

            model_inputs = BatchEncoding({'input_ids' : input_ids1,
                'attention_mask' : attention_mask1,
                'input_ids2' : input_ids2,
                'attention_mask2' : attention_mask2
            })


        if 'similar' in examples :
            model_inputs['labels'] = examples['similar']
        return model_inputs