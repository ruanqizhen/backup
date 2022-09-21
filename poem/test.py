from gen_text import GenText
from collections import Counter

generator = GenText()
result = generator.gen_poem("")
print(result)

# text = "春风吹落樱花雨|细雨洒开杨柳花"
# text = text.replace(' ', '')
# parts = text.split("|")
# if len(parts[0]) == len(parts[1]):
#     l = []
#     r = []
#     for i in range(len(parts[0])):
#         if parts[0][i] != parts[1][i]:
#             l.append(parts[0][i])
#             r.append(parts[1][i])
#
#     l_count = Counter(l)
#     r_count = Counter(r)
#     print(set(l) & set(r))
#     l_pattern = Counter(l_count.values())
#     r_pattern = Counter(r_count.values())

