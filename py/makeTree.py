import json
import argparse

class Node(object):
    def __init__(self):
        self.parent = None # Single object
        self.children = []  # Array of objects
        self.name = ""
        self.data = -1

    def add_node(self, name):
        for child in self.children:
            if child.name == name:
                return child
        new_child = Node()
        new_child.name = name
        new_child.parent = self
        self.children.append(new_child)
        return new_child

    def is_leaf(self):
        return not self.data == -1

    def __str__(self):
        if self.is_leaf() and len(self.children) == 0:
            # Actual Leaf
            p = '"name":"' + str(self.name) + '","id":' + str(self.data)
            #return p
        elif self.is_leaf() and len(self.children) > 0:
            # Node that has images connected to it but is a parent node.
            #pp = '"' + str(self.name) + '":' + str(self.data) + '},{' 
            #p = '"{name}":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
            p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
        else:
            # Part of hierarchy, no images attached
            p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
        return '{' + p + '}'

    def cull(self,drop,exc):
        if ((len(self.children) <= drop and len(self.children) > 0) or self.name in exc) and not self.is_leaf() and self.parent is not None:
            for idx,ch in enumerate(self.parent.children):
                if ch.name == self.name:
                    self.parent.children[idx] = self.children[0]
                    self.children[0].parent = self.parent
                    self.parent.children[idx].cull(drop,exc)
                    break
            for i in range(1,len(self.children)):
                self.parent.children.insert(i,self.children[i])
                self.children[i].parent = self.parent
                self.children[i].cull(drop,exc)
        elif len(self.children) > drop or self.parent is None:
            for ch in self.children:
                ch.cull(drop,exc)
        elif len(self.children) <= drop and len(self.children) > 0  and self.is_leaf():
            for ch in self.children:
                ch.cull(drop,exc)
        for ch1 in self.children:
            cnt = 0
            for ch2 in self.children:
                if ch1.name == ch2.name:
                    cnt += 1
            if cnt > 1:
                self.children.remove(ch1)

HELP_HIEARCHY = 'WordNet Hiearchy File'
HELP_ID_F = 'Distinct feature id file'
HELP_EXCLUDE = 'File containing words to remove. 1 word per line'
HELP_CULL = 'The amount of levels to cull from root'
parser = argparse.ArgumentParser()
parser.add_argument('--hierarchy_f', type=str, default='lsc_wn_hierarchy_t6.json')
parser.add_argument('--id_f', type=str, default='distinct_features_t6.json')
parser.add_argument('--exclude_f', type=str, default='')
parser.add_argument('--cull', type=int, default=-1)
args = parser.parse_args()

data = []

with open(args.hierarchy_f,'r') as f:
    data = json.load(f)

with open(args.id_f,'r') as f:
    ids = json.load(f)

exclude = set()
if args.exclude_f != '':
    with open(args.exclude_f,'r') as f:
        exclude = set([s.strip() for s in f.readlines()])


root = Node()

for idx,paths in enumerate(data):
    for path in paths:
        ch = root
        for p in path:
            ch = ch.add_node(p)
        ch.data = ids[idx]


root.name = "root"
root.data = -1
if args.cull != -1:
    root.cull(args.cull,exclude)
print(root)
