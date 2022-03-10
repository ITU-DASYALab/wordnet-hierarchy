import json

# Load the class descriptions for set features
#with open('12988_class_descriptions.txt','r') as f: #LCS
#with open('dralbumcovers/imgnet13k_classes.txt','r') as f:
with open('spotify/imgnet13k_classes.txt','r') as f:
    cls = [s.strip() for s in f.readlines()]

# Load the distinct top 6 features from all items (images)
#with open('distinct_features_t6.json', 'r') as f: #LSC
#with open('dralbumcovers/distinct_features_t25.json', 'r') as f:
with open('spotify/distinct_features_t25.json', 'r') as f:
    dcls = json.load(f)

# Get wordnet definitions of the features
dcls_def = [cls[i] for i in dcls]
split_cls = [(s.split(':')[0].strip().split(',')[0].strip().replace(' ','_'),s.split(':')[1].strip()) for s in dcls_def]

# Create file that only contains wordnet definitions
#with open('wn_distinct_t6.json','w') as f:
with open('spotify/wn_distinct_t25.json','w') as f:
    json.dump(split_cls,f)
