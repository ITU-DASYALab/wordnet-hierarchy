import json
import argparse

def breadcrumb_str(breadcrumb):
    path = ' -> '.join(list(map(lambda n: n.name, breadcrumb)))
    if path == "":
        path = 'root'
    return path

def cut_off_str(breadcrumb):
    bc = breadcrumb.copy()
    bc.reverse()
    path = ' <- '.join(list(map(lambda n: n.name, bc)))
    if path == "":
        path = 'root'
    return path

class Node(object):
    def __init__(self, name, parent):
        self.parent = parent # Single object
        self.children = []  # Array of objects
        self.name = name
        self.data = -1 # images link to this tag_id

    def add_node(self, name):
        for child in self.children:
            if child.name == name:
                return child
        new_child = Node(name, self)
        self.children.append(new_child)
        return new_child

    def num_children(self):
        return len(self.children)

    def is_tagged(self): #intermediate node, images link to tag
        #consider renaming: is_present_img_tag()
        return not self.data == -1

    def is_leaf_in_tree(self): #def has_no_children
        return len(self.children) == 0

    def has_one_child(self):
        return len(self.children) == 1

    def is_intermediate_node(self):
        return self.parent != None and not self.is_leaf_in_tree()
    
    def disconnect_from_parent(self):
        parent = self.parent
        parent.children.remove(self)
        self.parent = None
        return parent

    def clone(self, parent=None):
        node = Node(self.name, parent)
        node.data = self.data
        for ch in self.children:
            node.children.append(ch.clone(parent=node))
        return node

    def count_DFS(self, cmp):
        cnt = int(cmp(self)) 
        for ch in self.children:
            cnt += ch.count(cmp)
        return cnt

    def count_BFS(self, cmp):
        cnt = 0
        worklist = [self]
        while len(worklist) > 0:
            n = worklist.pop(0) # take first inserted element
            if cmp(n):
                cnt +=1
            for ch in n.children:
                worklist.append(ch)
        return cnt 

    def count_leafs(self):
        return self.count_BFS(lambda n: n.is_leaf_in_tree())

    def count_intermediate(self):
        return self.count_BFS(lambda n: n.is_intermediate_node())

    def count_only_child(self):
        return self.count_BFS(lambda n: n.has_one_child())

    def count_heigt(self):
        maxDepth = 0
        worklist = [ (self, 0) ]
        while len(worklist) > 0:
            n, depth = worklist.pop(0) #take first element 
            if depth > maxDepth:
                maxDepth = depth
            for ch in n.children:
                worklist.append((ch,depth+1))
        return maxDepth

    def count_widths(self):
        worklist = [ (self, 0) ]
        ret = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0) 
            ret[depth] = ret.get(depth, 0) + 1
            for ch in n.children:
                worklist.append((ch, depth+1))
        return ret
    
    def count_taggings(self, object_tag_relation):
        worklist = [ (self, 0) ]
        no_present_tag = missing_tagging = total = 0
        tags = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0)
            if n.data == -1:
                no_present_tag += 1
            elif n.data not in object_tag_relation:
                missing_tagging += 1
            else: 
                #print("tag: ", n.data, "\t#taggings: ",object_tag_relation[n.data])
                total += object_tag_relation[n.data]
                tags[depth] = tags.get(depth, 0) + object_tag_relation[n.data]
            for ch in n.children:
                worklist.append((ch, depth+1))
        return no_present_tag, missing_tagging, total, tags
    
    def avg_leaf_to_root(self):
        worklist = [ (self, 0) ]
        levels = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0) 
            levels[depth] = levels.get(depth, 0) + 1
            for ch in n.children:
                worklist.append((ch, depth+1))
        cnt_path_to_root = 0
        for n, children in levels.items():
            cnt_path_to_root += children*n
        total_nodes = self.count_BFS(lambda n: True)
        return cnt_path_to_root / total_nodes
    
    # can be extended to percentiles..
    def avg_child(self):
        worklist = [ (self, 0) ]
        levels = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0)
            depth_children = levels.get(depth)
            if not depth_children:
                depth_children = []
                levels[depth] = depth_children
            depth_children.append(len(n.children))
            for ch in n.children:
                worklist.append((ch, depth+1))
        avgs = {}
        for n , children in levels.items():
            avgs[n] = sum(children)/len(children)
        return avgs
    
    def has_child(self, wanted_name):
        for ch in self.children:
            if ch.name == wanted_name:
                return ch
        return None

    def lookup(self, name):
        worklist = [self]
        while len(worklist) > 0:
            n = worklist.pop(0) # take first inserted element
            if n.name == name:
                print("tag:\t", name,"\tfound at depth:\t", len(n.breadcrumb()), "\n#siblings\t", len(n.parent.children), "\t#children\t", len(n.children), "\n")
                return n
            for ch in n.children:
                worklist.append(ch)
                
    # go n levels down in orgTree
    # & returns a list of subTrees
    def split(self, target_depth, cut_offs=[]): #cut_from_top
        worklist = [ (self, 0) ]
        ret = []
        while len(worklist) > 0:
            n, depth = worklist.pop(0) #take first element 
            if depth == target_depth:
                ret.append(n)
                p = n.disconnect_from_parent()
                cut_offs.append(('split', n, p))
                #ret.append(n.clone()) #create new trees 
            elif depth < target_depth:
                for ch in n.children:
                    worklist.append((ch, depth+1))
        return ret
    
    def trim(self, target_depth, cut_offs=[]): #cut_from_bottom
        worklist = [ (self, 0) ]
        while len(worklist) > 0:
            n, depth = worklist.pop(0) #take first element 
            if depth == target_depth:
                cut_offs.append(('trim', n, n.parent)) # [ ab -> a ]
                n.disconnect_from_parent()
            elif depth < target_depth:
                for ch in n.children:
                    worklist.append((ch, depth+1))
                
    def absorbe(self, other, cut_offs=[], removed_nodes=None):
        if other.data != -1:
            cut_offs.append(('absorbe', other, other.breadcrumb(), other.breadcrumb_str()))
        for ch in other.children.copy(): # other.children changes
            self.steal_and_append_child(ch, cut_offs=cut_offs, removed_nodes=removed_nodes)
        other.parent = None #ready for garbage collection
        other.children = []
                
    def steal_and_append_child(self, ch, cut_offs=[], removed_nodes=None): #removes ch.node from curr position in tree and append here
        assert ch.parent is not None
        ch.parent.children.remove(ch)
        existing_ch = self.has_child(ch.name) #prevent dublicates
        if existing_ch: #merge
            if removed_nodes is not None:
                removed_nodes.store(ch, existing_ch)
            existing_ch.absorbe(ch, cut_offs=cut_offs, removed_nodes=removed_nodes)
        else: #insert
            if removed_nodes is not None:
                removed_nodes.store(ch, self)
            ch.parent = self
            self.children.append(ch)

    def collapse(self, cut_offs=[], removed_nodes=None): # do a check if there is imges attached
        assert self.is_intermediate_node()
        p = self.parent
        if removed_nodes is not None:
            removed_nodes.store(self, p)
        p.children.remove(self)
        p.absorbe(self, cut_offs=cut_offs, removed_nodes=removed_nodes)

    # initial collapse where dublicantes are not handled..
    def collapse_naive(self): # to do: check if there is imges attached
        assert self.is_intermediate_node()
        p = self.parent
        p.children.remove(self)
        for ch in self.children.copy(): #bug: can produce dublicate tags
            ch.parent = p
            p.children.append(ch)
        self.parent = None #ready for garbage collection
        self.children = []

    def visit(self, visitor, *args, **kwargs):
        worklist = [ self ]
        while len(worklist) > 0:
            n = worklist.pop(0)
            visitor(n, *args, **kwargs)
            for ch in n.children:
                worklist.append(ch)

    def __str__(self):
        if self.is_tagged() and len(self.children) == 0:
            #actual leaf
            p = '"name":"' + str(self.name) + '","id":' + str(self.data)
            #return p
        elif self.is_tagged() and len(self.children) > 0:
            # node with images & and is also a parent
            #pp = '"' + str(self.name) + '":' + str(self.data) + '},{' 
            #p = '"{name}":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
            p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
        else:
            # part of hierachy, no images attached
            p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
        return '{' + p + '}'

    def cull(self,drop,exc): #drop er interval constant , exc -> exclusive
        if ((len(self.children) <= drop and len(self.children) > 0) or self.name in exc) and not self.is_tagged() and self.parent is not None:
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
        elif len(self.children) > drop or self.parent is None: #root eller flere børn end drop
            for ch in self.children:
                ch.cull(drop,exc)
        elif len(self.children) <= drop and len(self.children) > 0  and self.is_tagged(): #like top case, include leaf
            for ch in self.children:
                ch.cull(drop,exc)
        for ch1 in self.children: #dublicates ...
            cnt = 0
            for ch2 in self.children:
                if ch1.name == ch2.name:
                    cnt += 1
            if cnt > 1: #
                self.children.remove(ch1)
    
    def printTree(self, bend=0):
        print(" "*bend + self.name + "\tkids: " + str(len(self.children)))
        #print(" "*bend + self.name + "\tid: " + str(self.data) + "\tkids: " + str(len(self.children)))
        for i in self.children:
            i.printTree(bend+1)

    def printGraphNode(self, f, depth):
        own_breadcrumb = self.breadcrumb_str()
        f.write(f'"{own_breadcrumb}" [label="{self.name}"]\n') #define node
        if depth < 0:
            return
        for ch in self.children:
            ch.printGraphNode(f, depth-1)
                        
    def printGraph(self, f, depth):
        own_breadcrumb = self.breadcrumb_str()
        if depth < 0:
            return
        for ch in self.children:
            f.write(f'"{own_breadcrumb}" -> "{ch.breadcrumb_str()}"\n')
            ch.printGraph(f, depth-1)

    def root(self):
        n = self
        while n.parent:
            n = n.parent
        return n

    def breadcrumb(self):
        path = []
        n = self
        while n:
            path.append(n)
            n = n.parent
        path.reverse()
        return path

    def breadcrumb_str(self):
        return breadcrumb_str(self.breadcrumb())

def commonWords(Node):
    # input: list of common words
    # exclude advance search terms
        # possible extend to domain specific scientific words
    return

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

with open(args.id_f,'r') as f:
    ids = json.load(f)

def load_data_1():
    data = []
    with open(args.hierarchy_f,'r') as f:
        data = json.load(f)
        
    exclude = set()
    if args.exclude_f != '':
        with open(args.exclude_f,'r') as f:
            exclude = set([s.strip() for s in f.readlines()])
    tree = Node("root",None)

    for idx,paths in enumerate(data):
        for path in paths:
            ch = tree
            for p in path:
                ch = ch.add_node(p)
            ch.data = ids[idx]
    tree.data = -1
    if args.cull != -1:
        tree.cull(args.cull,exclude) 
    return tree

def from_json_to_tree(tree, parent=None) :
    node = Node(tree['name'], parent)
    node.data = tree['id']
    if 'children' in tree:
        for i in tree['children']:
            node.children.append(from_json_to_tree(i, parent=node))
    return node

def load_data_2():
    with open('jsTree/tree_t25.json', 'r') as f:
    #with open('jsTree/tree_t5c0.json', 'r') as f:
    #with open('new_org.json', 'r') as f:
        tree = json.load(f)
        #print( from_json_to_tree(tree) )
    return from_json_to_tree(tree)

root = load_data_2()

#root.printTree()

# budget is the max number of children a node can have
# merge all grandchildren at once. If not possible -> dont merge
def collapse_nodes(n, budget=5, cut_offs=[], removed_nodes=None):
    #print('> ', n.name, budget)
    rerun = True
    while rerun:
        rerun = False
        candidates = {}
        sum = 0
        for ch in n.children:
            if ch.is_intermediate_node():
                candidates[ch] = ch.num_children()
                sum += ch.num_children()
            else:
                sum += 1 #leaf
        if sum <= budget:
            worklist = n.children.copy() # n.children is being modified during while loop
            for ch in worklist:
                if ch.is_intermediate_node():
                    ch.collapse(cut_offs=cut_offs, removed_nodes=removed_nodes)
                    rerun = True
                # else is_leaf() -> do nothing

# merge node with fewest children first. This gives rare tags an advantage
def collapse_nodes_unfair(n, budget=5):
    #print('> ', n.name, budget)
    rerun = True
    while rerun:
        rerun = False
        candidates = {}
        left = budget - n.num_children()
        for ch in n.children:
            if ch.is_intermediate_node():
                candidates[ch] = ch.num_children()
        for ch, cnt in sorted(candidates.items(), key=lambda t: t[1]): #sort by fewest childs
            if cnt <= 1 or cnt - 1 <= left:
                left -= cnt - 1
                ch.collapse()
                rerun = True

def collapse_single_child(n):
    n.visit(lambda n : n.collapse() if n.is_intermediate_node() and n.has_one_child() else ())

def write_graph(tree, filename, depth):        
    with open(filename, 'w+') as f: #w+ creates file if it does not exist
        f.write("digraph {\n")
        f.write("{\n")
        tree.printGraphNode(f, depth)
        f.write("}\n")
        tree.printGraph(f, depth)
        f.write("}\n")
        f.close()
        
def write_json(tree, filename):        
    with open(filename, 'w+') as f:
        f.write(str(tree))
        f.close()

def write_tsv(combined_dict, filename):
    keys = "\t".join(['Level','#Nodes','#Taggings'])
    with open(filename, 'w+') as f:
        f.write(keys+"\n")
        for k , v in combined_dict.items():
            #print(combined_dict[k])
            values = "\t".join(map(str, combined_dict[k]))
            f.write(str(k)+"\t"+values+"\n")
        f.close()
    
class NodeHistory:
    def __init__(self):
        self.history = {}

    def __len__(self):
        return len(self.history.items())
    
    def store(self, src, dst):
        self.history[src.breadcrumb_str()] = (dst.breadcrumb(), src.name)
        
    def resolve(self, bc):
        bcstr = breadcrumb_str(bc)
        while bcstr in self.history: 
            bc, step_name = self.history[bcstr] 
            bcstr = breadcrumb_str(bc)
        return bc
    

def write_cut_offs_absorbe(elem, f, removed_nodes=None, *args, **kwargs):
    _, ch, breadcrumb, orig_bcstr = elem
    assert breadcrumb_str(breadcrumb) == orig_bcstr
    assert removed_nodes is not None
    final_bc = removed_nodes.resolve(breadcrumb)
    #print("leaf: ", ch.name, "\n", breadcrumb_str(breadcrumb))
    #print(breadcrumb_str(final_bc))
    final_parent = final_bc[len(final_bc)-1]
    if final_parent.name == ch.name:
        final_parent = final_bc[len(final_bc)-2]
    final_root = final_parent.root()
    if ch.data != -1:
        f.write(f'{final_root.name}:{final_root.name}:{final_parent.name}:{ch.name}\n')
        
def write_cut_offs_trim(elem, f, *args, **kwargs):
    _, ch, p = elem
    def visitor(n):
        if n.data != -1:
            f.write(f'{p.root().name}:{p.root().name}:{p.name}:{n.name}\n')
    ch.visit(visitor)
  
def write_cut_offs_split(elem, f, *args, **kwargs):
    _, ch, p = elem
    print('split: ', ch.name, " -> ", cut_off_str(p.breadcrumb()))
    
def write_cut_offs_verbose_absorbe(elem, f, removed_nodes=None, *args, **kwargs):
    _, ch, breadcrumb, orig_bcstr = elem
    assert breadcrumb_str(breadcrumb) == orig_bcstr
    assert removed_nodes is not None
    final_bc = removed_nodes.resolve(breadcrumb)
    final_parent = final_bc[len(final_bc)-1]
    if final_parent.name == ch.name:
        final_parent = final_bc[len(final_bc)-2]
    final_root = final_parent.root()
    f.write(f'ABSORBE: {final_parent.breadcrumb_str()} ==> {ch.name} | data: {ch.data}\n')

def write_cut_offs_verbose_trim(elem, f, *args, **kwargs):
    _, ch, p = elem
    def visitor(n):
        p_bc_str = p.breadcrumb_str()
        cutoff = n.breadcrumb_str()
        f.write(f'TRIM: {p_bc_str} ==> {cutoff} | data: {n.data}\n')
    ch.visit(visitor)

## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)
write_cut_offs_types = {
    'absorbe': write_cut_offs_absorbe,
    'trim': write_cut_offs_trim,
    #'split': write_cut_offs_split,
}

write_cut_offs_verbose_types = {
    'absorbe': write_cut_offs_verbose_absorbe,
    'trim': write_cut_offs_verbose_trim,
    #'split': write_cut_offs_verbose_split,
}

def write_cut_offs(cut_offs, filename, handlers=write_cut_offs_types, *args, **kwargs):
    with open(filename, 'w+') as f:
        for elem in cut_offs:
            elem_type = elem[0]
            handler = handlers.get(elem_type)
            if handler:
                handler(elem, f, *args, **kwargs)
        f.close()

def combine_dicts(dict1, dict2):
    dict_list = [dict1, dict2]            
    from collections import defaultdict
    #defaultdict does not require consistent keys across dictionaries, and  maintains O(n) time 
    d = defaultdict(list)
    for mydict in dict_list:
        for k, v in mydict.items():
            #print(k, v)
            d[k].append(v)
    return d

def read_object_tag_relation(filename):
    object_tag_relation = {}
    with open(filename) as file:
        file.readline() #trow away first line
        for line in file:
            (key, val) = line.split(",")
            object_tag_relation[int(key)] = int(val)
    return object_tag_relation

def print_stats(tree, tsvfile):
    print("total nodes:\t",tree.count_BFS(lambda n: True))
    print("leafs:\t\t\t", tree.count_leafs())
    print("intermediate:\t", tree.count_intermediate())
    print("only child:\t\t", tree.count_only_child())
    print("height:\t\t\t", tree.count_heigt())
    print("(level: avg child)")
    print(tree.avg_child())
    print("(level: #child)")
    print(tree.count_widths())
    object_tag_relation = read_object_tag_relation("/home/ek/Documents/Thesis/SQL/tag_count.csv")
    no_present_tag , missing_tagging, total, tag_dict = tree.count_taggings(object_tag_relation)
    print("tag '-1': ", no_present_tag)
    print("not in object_tag_relation: ",missing_tagging)
    print("total found tags: ", total)
    
    comb_dict = combine_dicts(tree.count_widths(), tag_dict)
    write_tsv(comb_dict, tsvfile)
    
    print("avg leaf to root: ", tree.avg_leaf_to_root())
    print()
    #avg (children) / level -> tree.avg_child()
    
#foo = root.clone().lookup("animal")
#root.lookup("animal")
#print_stats(foo)
#print(' -> '.join(list(map(lambda n: n.name, foo.breadcrumb()))), "\n")

print("UNCOMPRESSED")
print_stats(root, 'stats_before_collapse.tsv')

cut_offs=[]
removed_nodes= NodeHistory()
root.visit(collapse_nodes, budget=20, cut_offs=cut_offs, removed_nodes=removed_nodes)
collapse_single_child(root)
print_stats(root, 'stats_after_collapse.tsv')
print("cut_offs before split from top: ", len(cut_offs))

#write_json(root, 'new_org_optimized_max20.json')
#write_graph(root, 'graph.dot', 2)
    
subtree_list1 = root.split(1, cut_offs=cut_offs)
if removed_nodes is not None:
    print("cut_offs: ", len(cut_offs), " removed_nodes: ", len(removed_nodes))
for subtree in subtree_list1:
    subtree.trim(7, cut_offs=cut_offs)

print("cut_offs: ", len(cut_offs) )

#default
write_cut_offs(cut_offs, 'cut_off.txt', removed_nodes=removed_nodes)
#debug
write_cut_offs(cut_offs, 'cut_off_verbose.txt', handlers=write_cut_offs_verbose_types, removed_nodes=removed_nodes)
