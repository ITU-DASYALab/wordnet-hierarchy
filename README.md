# wordnet-hierarchy
Scripts to get the WordNet hierarchy

## Requirements

Python 3.7 or higher

nltk package

```
pip3 install nltk
```

## py/makeWordNetDef.py

This file is used to get the WordNet definitions from the 12988 features.
It requires the class descriptions file along with a file containing the distinct features to get definitions from.

## py/makeHierarchy.py

This uses the output file from makeWordNetDef.py to get the hierarchy path to each word

## py/makeTree.py

This file creates a tree structure using the output file from makeHierarchy.py and the distinct feature file. It can take a file to remove specific nodes from the tree, and a number that determines how many levels from the root to cull, default being none.
