import re

class Normalizer :

    def __init__(self, ) :
        pass

    def normalize_string(self, code) :
        code = re.sub('\".{2,}\"', '"..."', code)
        code = re.sub('\'.{2,}\'', '"..."', code)
        return code

    def normalize_character(self, code) :
        code = re.sub('\".{1}\"', "'.'", code)
        code = re.sub("\'.{1}\'", "'.'", code)
        return code

    def normalize_numeric(self, code) :
        code = re.sub('[0-9]+', '1', code)
        code = re.sub('^(1\.)[0-9]+', '1.0', code)
        return code

    def extract_vars(self, code) :
        code = re.sub('=', ' = ', code)
        code = re.sub('=  =', '==', code)
        vars = re.findall('[0-9a-zA-Z_, ]*(?=[^+-/%*=<>]=)', code)

        vars = [v.strip() for v in sum([v.split(',') for v in vars], [])]
        vars = [v for v in vars if v != '' and ' ' not in v]
        return list(set(vars))

    def change_vars(self, code, vars) :
        for i, v in enumerate(vars) :
            dummy = f"P{i}"

            regex_tar = '(?=' + v + '[^a-z0-9_])'
            tmp = re.sub(regex_tar, dummy, code)

            regex_roll = '[a-z]' + dummy + v
            rollbacks = list(set(re.findall(regex_roll,  tmp)))
            
            for r in rollbacks :
                tmp = re.sub(r, r[0] + r[-1], tmp)

            tmp = re.sub(dummy + v, dummy, tmp)
            code = tmp
            
        return code

    def normalize(self, code) :
        code = self.normalize_string(code)
        code = self.normalize_character(code)
        code = self.normalize_numeric(code)
        
        variables = self.extract_vars(code)
        code = self.change_vars(code, variables)
        return code

    def __call__(self, datasets) :
        code1_list = []
        code2_list = []

        size = len(datasets['code1'])
        for i in range(size) :
            code1 = self.normalize(datasets['code1'][i])
            code2 = self.normalize(datasets['code2'][i])
            
            code1_list.append(code1)
            code2_list.append(code2)

        datasets['code1'] = code1_list
        datasets['code2'] = code2_list
        return datasets