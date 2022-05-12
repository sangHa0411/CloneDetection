import re

class Normalizer :

    def __init__(self,) :
        pass

    def extract_vars(self, code) :
        code = re.sub('\(.+\)', '()', code)
        vars = re.findall('[0-9a-zA-Z_, ]+(?==[^+-/=%*<>])', code)

        vars = [v.strip() for v in sum([v.split(',') for v in vars], [])]
        vars = [v for v in vars if v != '' and ' ' not in v]
        return list(set(vars))

    def change_vars(self, code, vars) :
        for i, v in enumerate(vars) : 
            dummy = f'P{i}'
            code = re.sub('(?<![a-zA-Z0-9])(?=' + v + '[^a-zA-Z0-9_])', dummy, code)
            code = re.sub(dummy + v, dummy, code)
        return code

    def change(self, code) :
        vars = self.extract_vars(code)
        code = self.change_vars(code, vars)
        return code

    def __call__(self, datasets) : 
        code1_list = []
        code2_list = []

        size = len(datasets['code1'])
        for i in range(size) :
            code1_list.append(self.change(datasets['code1'][i]))
            code2_list.append(self.change(datasets['code2'][i]))

        datasets['code1'] = code1_list
        datasets['code2'] = code2_list
        return datasets