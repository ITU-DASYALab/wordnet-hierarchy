from nltk.corpus import wordnet as wn
import json

# Load the distinct wordnet definitions
wnd = []
with open('spotify/wn_distinct_t25.json', 'r') as f:
    wnd = json.load(f)

# Get hierarchy
hierarchy = []
# s1 = Word, s2 = Definition
for s1,s2 in wnd:
    h = []
    # Find the synset matching the description of s1
    for syn in wn.synsets(s1):
        if s2.split(';')[0] in syn.definition() and '.n.' in syn.name():
            # Get the hierarchy to s1
            for hyp in syn.hypernym_paths():
                hh = []
                for path in hyp:
                    hh.append(path.lemma_names()[0])
                h.append(hh)
    hierarchy.append(h)

# List elements that failed
for idx,h in enumerate(hierarchy):
    if len(h) == 0:
        print('FAILED', idx)

# Store the hierarchy
with open('spotify/wn_hierarchy_t25.json','w') as f:
    json.dump(hierarchy,f)
