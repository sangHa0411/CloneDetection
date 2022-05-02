

class BlockDeleter : 
    def __init__(self, ) :
        pass

    def search(self, sen_list, string) :
        for i,sen in enumerate(sen_list) :
            if string in sen :
                return i
        return -1

    def delete_annotation_block(self, code, string) :
        sens = [sen for sen in code.split('\n')]

        start_id = self.search(sens, string)
        end_id = self.search(sens[start_id+1:], string)
        if end_id != -1 :
            end_id += (start_id+1)
            code = sens[:start_id] + sens[end_id+1:]
        else :
            code = sens[:start_id] + sens[start_id+1:]

        code = '\n'.join(code)
        return code

    def delete_block(self, code, string) :
        while string in code :
            code = self.delete_annotation_block(code, string)
        return code

    def __call__(self, code) :
        code = self.delete_block(code, '"""')
        code = self.delete_block(code, "'''")
        return code


class Preprocessor :
    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer 

    def delete_annotation(self, code) :
        sens = code.split('\n')

        sens_processed = []
        for sen in sens :
            if '#' in sen :
                index = sen.index('#')
                sen = sen[:index]
            sens_processed.append(sen)

        return '\n'.join(sens_processed)

    def split(self, code, tokenizer) :
        sens = code.split('\n')
        sens = [sen.strip() for sen in sens if sen.strip() != '']
        return tokenizer.sep_token.join(sens)

    def __call__(self, code) :        
        code = self.delete_annotation(code) 
        code_input = self.split(code, self.tokenizer)
        return code_input