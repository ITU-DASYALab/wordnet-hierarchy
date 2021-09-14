import json

# As duplicate definitions exists we collect and store those here

d = []
with open('wn_distinct_t6.json','r') as f:
    d = json.load(f)

dup = []
for idx,w in enumerate(d):
    if w in dup:
        continue
    is_dup = False
    for idx2,ww in enumerate(d):
        if w[0] == ww[0] and idx != idx2:
            dup.append(ww)
            is_dup = True
    if is_dup:
            dup.append(w)


dup_dict = {}
for i in dup:
    if i[0] in dup_dict:
            dup_dict[i[0]].append(i[1])
    else:
            dup_dict[i[0]] = []
            dup_dict[i[0]].append(i[1])

with open('duplicates.json','w') as f:
    json.dump(dup,f)

with open('duplicates_dict.json','w') as f:
    json.dump(dup_dict,f)
