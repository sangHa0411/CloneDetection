import re

class Preprocessor : 
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

    def delete_annotation(self, code) :
        sens = code.split('\n')

        sens_processed = []
        for sen in sens :
            if '#' in sen :
                index = sen.index('#')
                sen = sen[:index]
            sens_processed.append(sen)

        return '\n'.join(sens_processed)

    def preprocess(self, code) :
        code = self.delete_block(code, '"""')
        code = self.delete_block(code, "'''")
        code = self.delete_annotation(code)
        return re.sub('\s+', ' ', code)

    def __call__(self, datasets) :
        code1_list = []
        code2_list = []

        size = len(datasets['code1'])
        for i in range(size) :
            code1 = self.preprocess(datasets['code1'][i])
            code2 = self.preprocess(datasets['code2'][i])
            
            code1_list.append(code1)
            code2_list.append(code2)

        datasets['code1'] = code1_list
        datasets['code2'] = code2_list
        return datasets