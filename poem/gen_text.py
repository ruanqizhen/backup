from transformers import AutoTokenizer, GPT2LMHeadModel
from collections import Counter


class GenText:
    def __init__(self):
        model_path = "./model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).cuda()

    def gen_couplet(self, shang_lian: str):
        if shang_lian and shang_lian.strip():
            shang_lian = shang_lian.strip()
            if shang_lian[-1] in ['；', '。']:
                shang_lian = shang_lian[:-1]
            shang_lian += "|"
        else:
            shang_lian = ""

        seed = self.tokenizer(shang_lian, return_tensors="pt").input_ids.cuda()
        seed = seed[:, :-1]
        output = self.model.generate(seed, do_sample=True,
                                     max_length=128, temperature=0.7, num_return_sequences=8, repetition_penalty=1.1)
        result = [self.tokenizer.decode(x, skip_special_tokens=True) for x in output]
        return self.cleanup_couplet(result, shang_lian)

    @staticmethod
    def cleanup_couplet(raw, shang_lian):
        results = []
        for text in raw:
            if "|" in text and (len(text) > 16 or shang_lian):
                text = text.replace(' ', '')
                parts = text.split("|")
                l_part = parts[0]
                r_part = parts[1]
                if len(l_part) == len(r_part):
                    l_chars = []
                    r_chars = []
                    for i in range(len(l_part)):
                        if l_part[i] != r_part[i]:
                            l_chars.append(l_part[i])
                            r_chars.append(r_part[i])
                    if set(l_chars) & set(r_chars):
                        continue
                    l_count = Counter(list(l_part))
                    r_count = Counter(list(r_part))
                    l_pattern = Counter(l_count.values())
                    r_pattern = Counter(r_count.values())
                    if l_pattern == r_pattern:
                        results.append(text.replace('|', '；'))
        return list(set(results))

    def gen_poem(self, title: str):
        if title and title.strip():
            title = title.strip() + ">"
        else:
            title = ">"

        seed = self.tokenizer(title, return_tensors="pt").input_ids.cuda()
        seed = seed[:, :-1]
        output = self.model.generate(seed, do_sample=True,
                                     max_length=128, temperature=0.7, num_return_sequences=6, repetition_penalty=1.2)
        result = [self.tokenizer.decode(x, skip_special_tokens=True) for x in output]
        return self.cleanup_poem(result)

    @staticmethod
    def cleanup_poem(raw):
        results = []
        for text in raw:
            text = text.replace(' ', '')
            text = text.split(">", 1)[1]
            if len(text) != 24 and len(text) != 32 and len(text) != 48 and len(text) != 64:
                continue
            words = text.replace("，", "").replace("。", "")
            counter = Counter(words)
            if counter.most_common(1)[0][1] > 2:
                continue
            results.append(text)
        return results
