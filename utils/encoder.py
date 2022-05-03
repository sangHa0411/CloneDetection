
class Encoder :
    def __init__(self, tokenizer, similarlity_flag: bool, max_input_length: int) :
        self.tokenizer = tokenizer
        self.similarlity_flag = similarlity_flag
        self.max_input_length = max_input_length
    
    def __call__(self, examples):
        if self.similarlity_flag :
            model_inputs = self.tokenizer(examples['code1'], 
                max_length=self.max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            model_inputs2 = self.tokenizer(examples['code2'], 
                max_length=self.max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

            model_inputs['input_ids2'] = model_inputs2['input_ids']
            model_inputs['attention_mask2'] = model_inputs2['attention_mask']
        else :
            model_inputs = self.tokenizer(examples['code1'], 
                examples['code2'],
                max_length=self.max_input_length, 
                return_token_type_ids=False,
                truncation=True
            )

        if 'similar' in examples :
            model_inputs['labels'] = examples['similar']
        return model_inputs