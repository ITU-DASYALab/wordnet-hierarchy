import json
import argparse
from os import posix_fallocate

# return a string representation of path in tree
def breadcrumb_str(breadcrumb):
    path = ' -> '.join(list(map(lambda n: n.name, breadcrumb)))
    if path == "":
        path = 'root'
    return path

# return a string representation of cut offs. Used for cutting tree from top
def cut_off_str(breadcrumb):
    bc = breadcrumb.copy()
    bc.reverse()
    path = ' <- '.join(list(map(lambda n: n.name, bc)))
    if path == "":
        path = 'root'
    return path

collapse_cnt = 0

# encapsules the concept of a node in the tree
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

    def is_leaf_in_tree(self): #definition of leaf: node that has no children
        return len(self.children) == 0

    def has_one_child(self):
        return len(self.children) == 1
    
    def is_only_child(self):
        p = self.parent
        if p is None:
            return False
        return p.has_one_child()

    def is_intermediate_node(self):
        return self.parent != None and not self.is_leaf_in_tree()
    
    # detects if a node is part of singular chain and shold be collapsed
    def should_collapse_singular_chain(self):
        is_only_child_of_root = self.is_only_child() and self.parent is None
        return self.is_intermediate_node() and (self.has_one_child() or is_only_child_of_root)
        
    # remove node from current position
    def disconnect_from_parent(self):
        parent = self.parent
        parent.children.remove(self)
        self.parent = None
        return parent

    # recursively clones the tree
    def clone(self, parent=None):
        node = Node(self.name, parent)
        node.data = self.data
        for ch in self.children:
            node.children.append(ch.clone(parent=node))
        return node

    # count using depth first search
    def count_DFS(self, cmp):
        cnt = int(cmp(self)) 
        for ch in self.children:
            cnt += ch.count(cmp)
        return cnt

    # count using breadth first search
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
    
    def count_collapsable_singular_chain(self):
        return self.count_BFS(lambda n: n.should_collapse_singular_chain())
    
    # traverses the tree and apply visitor function
    def visit(self, visitor, *args, **kwargs):
        worklist = [ self ]
        while len(worklist) > 0:
            n = worklist.pop(0)
            visitor(n, *args, **kwargs)
            for ch in n.children:
                worklist.append(ch)
                
    # duplicates detection, cnt occurances
    def nodes_group_by(self, selector):
        ret = {}
        def visitor(n):
            key = selector(n)
            entries = ret.get(key)
            if entries is None:
                entries = []
                ret[key] = entries
            entries.append(n)
        self.visit(visitor)
        return ret
    
    def find_duplicates(self, selector):
        dups = self.nodes_group_by(selector)
        return {k: v for k, v in dups.items() if len(v) >= 2} #filtering dict

    def find_duplicates_by_name(self):
        return self.find_duplicates(lambda n: n.name)

    def find_duplicates_by_data(self):
        return self.find_duplicates(lambda n: n.data)

    # check if subtrees in two trees are identical
    def compare_tree(self, other):
        worklist = [ (self, other) ]
        while len(worklist) > 0:
            a, b = worklist.pop(0)
            if a.name != b.name:
                return False
            if a.data != b.data:
                return False
            if len(a.children) != len(b.children):
                return False
            seen_a = {}
            for ch in a.children:
                seen_a[ch.name] = ch
            for ch in b.children:
                if not ch.name in seen_a:
                    return False
                worklist.append( (seen_a[ch.name], ch) ) # appending children to future work
        return True

    # count max height of tree
    def count_height(self):
        maxDepth = 0
        worklist = [ (self, 0) ]
        while len(worklist) > 0:
            n, depth = worklist.pop(0) #take first element 
            if depth > maxDepth:
                maxDepth = depth
            for ch in n.children:
                worklist.append((ch,depth+1))
        return maxDepth

    # count width at all levels in tree. Returns a dictionary
    def count_widths(self):
        worklist = [ (self, 0) ]
        ret = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0) 
            ret[depth] = ret.get(depth, 0) + 1
            for ch in n.children:
                worklist.append((ch, depth+1))
        return ret
    
    # count number of taggings at all levels in the tree. Returns a dictionary
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
    
    # count the average height from leaf to root
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
    
    # count average number of children at all levels in the tree. Returns a dictionary
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
                
    # goes n levels down in the org-tree and returns a list of sub-trees
    # cut_from_top
    def split(self, target_depth, cut_offs=[]):
        worklist = [ (self, 0) ]
        ret = []
        while len(worklist) > 0:
            n, depth = worklist.pop(0) 
            if depth == target_depth:
                ret.append(n)
                #ret.append(n.clone()) #create new trees 
                p = n.disconnect_from_parent()
                cut_offs.append(('split', n, p))
            elif depth < target_depth:
                for ch in n.children:
                    worklist.append((ch, depth+1))
        return ret
    
    # goes n levels down and cuts off all nodes below this level
    # cut_from_bottom
    def trim(self, target_depth, cut_offs=[]):
        worklist = [ (self, 0) ]
        while len(worklist) > 0:
            n, depth = worklist.pop(0) #take first element 
            if depth == target_depth:
                cut_offs.append(('trim', n, n.parent)) # [ ab -> a ]
                n.disconnect_from_parent()
            elif depth < target_depth:
                for ch in n.children:
                    worklist.append((ch, depth+1))

    # check is a node has a certain child. Returns the child if found. Used for dublicate detection.
    def has_child(self, wanted_name):
        for ch in self.children:
            if ch.name == wanted_name:
                return ch
        return None

    # parent node absorbe collapsed node
    def absorb(self, other, cut_offs=[], removed_nodes=None):
        #print(f'  --absorbe-- self: {self.name} other: {other.name}')
        if other.data != -1: # do a check if there is images attached
            cut_offs.append(('absorbe', other, other.breadcrumb(), other.breadcrumb_str()))
        for ch in other.children.copy(): # other.children changes
            self.steal_and_append_child(ch, cut_offs=cut_offs, removed_nodes=removed_nodes)
        other.parent = None #ready for garbage collection
        other.children = []

    # removes childs from current position in tree and append to new parent
    def steal_and_append_child(self, ch, cut_offs=[], removed_nodes=None):
        assert ch.parent is not None
        #print(f'  --steal & append-- self: {self.name} ch: {ch.name}')
        ch.parent.children.remove(ch)
        existing_ch = self.has_child(ch.name) #prevent dublicates
        if existing_ch: #merge
            if removed_nodes is not None:
                removed_nodes.store(ch, existing_ch)
            existing_ch.absorb(ch, cut_offs=cut_offs, removed_nodes=removed_nodes)
        else: #insert
            if removed_nodes is not None:
                removed_nodes.store(ch, self)
            ch.parent = self
            self.children.append(ch)

    # initializes the collapse of a node
    def collapse(self, cut_offs=[], removed_nodes=None): 
        #global collapse_cnt
        #collapse_cnt += 1
        #my_cnt = collapse_cnt
        #self.root().print_tree()
        #print(f'--collapse {my_cnt}-- self: {self.name} ')
        assert self.is_intermediate_node()
        p = self.parent
        if removed_nodes is not None:
            removed_nodes.store(self, p)
        p.children.remove(self)
        p.absorb(self, cut_offs=cut_offs, removed_nodes=removed_nodes)
        #self.root().print_tree()
        #print(f'--collapse {my_cnt}-- self: {self.name} ')
    
    # initial collapse where dublicantes are not handled well
    def collapse_naive(self): # to do: check if there is imges attached
        assert self.is_intermediate_node()
        p = self.parent
        p.children.remove(self)
        for ch in self.children.copy(): #bug: can produce dublicate tags
            ch.parent = p
            p.children.append(ch)
        self.parent = None #ready for garbage collection
        self.children = []
        
    # not done yet ..
    def commonWords(self, words=[]):
        # input: list of common words
        # exclude advance search terms
            # possible extend to domain specific scientific words
        return

    # to string
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

    # find a certain tag in the tree. Returns the node and print some statistics
    def lookup(self, name):
        worklist = [self]
        while len(worklist) > 0:
            n = worklist.pop(0) # take first inserted element
            if n.name == name:
                print("tag:\t", name,"\tfound at depth:\t", len(n.breadcrumb()), "\n#siblings\t", len(n.parent.children), "\t#children\t", len(n.children), "\n")
                return n
            for ch in n.children:
                worklist.append(ch)

    # print tree using offset. Intermediate nodes are printed with their count of children
    def print_tree(self, bend=0):
        bendstr = " "*bend
        if self.is_leaf_in_tree():
            print(bendstr, self.name, self.data)
        else:
            print(bendstr, self.name, self.data, "\t\tkids: ", str(len(self.children)), "\tsize subtree:", self.count_BFS(lambda n: True))
        for i in self.children:
            i.print_tree(bend+1)

    # prints a node in the format of graphviz
    def printGraphNode(self, f, depth):
        own_breadcrumb = self.breadcrumb_str()
        f.write(f'"{own_breadcrumb}" [label="{self.name}"]\n') #define node
        if depth < 0:
            return
        for ch in self.children:
            ch.printGraphNode(f, depth-1)

    # prints graph in format of graphviz
    def printGraph(self, f, depth):
        own_breadcrumb = self.breadcrumb_str()
        if depth < 0:
            return
        for ch in self.children:
            f.write(f'"{own_breadcrumb}" -> "{ch.breadcrumb_str()}"\n')
            ch.printGraph(f, depth-1)

    # find the root node of a given node
    def root(self):
        n = self
        while n.parent:
            n = n.parent
        return n

    # return path of nodes in the tree
    def breadcrumb(self):
        path = []
        n = self
        while n:
            path.append(n)
            n = n.parent
        path.reverse()
        return path

    # returns a string representaion of breadcrumb
    def breadcrumb_str(self):
        return breadcrumb_str(self.breadcrumb())

# An object to keep track of all removed nodes.
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

# loads data from args
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

# turns a json tree into an object tree recursively
def from_json_to_tree(tree, parent=None) :
    node = Node(tree['name'], parent)
    node.data = tree['id']
    if 'children' in tree:
        for i in tree['children']:
            node.children.append(from_json_to_tree(i, parent=node))
    return node

# loads data from file
def load_data_2():
    #with open('jsTree/test_twochild.json', 'r') as f:
    with open('jsTree/tree_t5c0.json', 'r') as f:
    #with open('new_org.json', 'r') as f:
        tree = json.load(f)
        #print( from_json_to_tree(tree) )
    return from_json_to_tree(tree)

root = load_data_2()

#root.print_tree()

# merge all grandchildren at once. If all gr_ch doesn't fit in the budget we don't merge.
# budget is the max number of children a node can have.
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

# collapse all nodes that is a one_child and has one_child
def collapse_all_single_children(n):
    i = 0 
    while n.count_collapsable_singular_chain() > 0:
        i += 1
        #pre_cnt = n.count_collapsable_singular_chain()
        n.visit(lambda n : n.collapse() if n.should_collapse_singular_chain() else ())
        #* can change #leafes -> because of collapse..
        #post_cnt = n.count_collapsable_singular_chain()
        #print(f'i: {i} cnt: {pre_cnt} -> {post_cnt} change: {pre_cnt-post_cnt}')
        assert i < 100 #safety feature to avoid eternal loop is non solvable due to a bug..

# split tree from top
def split_automated(tree, minNodes=50, maxNodes=500, subtrees=[]):
    rerun = True
    while rerun:
        rerun = False
        split = tree.split(1, cut_offs=cut_offs)
        subtrees.extend(split) # add all trees to working list
        for elem in subtrees:
            cnt = elem.count_BFS(lambda n: True)
            if cnt > maxNodes:
                subtrees.remove(elem)
                new_subtrees = elem.split(1, cut_offs=cut_offs)
                #print(f'MAX:: {elem.name}\t size: {cnt}\t new_subtrees: {len(new_subtrees)}')
                subtrees.extend(new_subtrees)
                rerun = True
            if cnt < minNodes:
                #elem.print_tree()
                #print()
                subtrees.remove(elem)
                rerun = True
    return subtrees

#print top 10 most frequent
def print_top10_dup(dups):
    i = 0
    for k in sorted(dups, key=lambda k: min([len(n.breadcrumb())] for n in dups[k])): # sort by top most duplicate entry
    #for k in sorted(dups, key=lambda k: len(dups[k]), reverse=True): # sort by value
        entries = dups[k]
        i += 1
        if i > 10:
            break
        print(k, entries[0].name,len(entries))
        
# count sum of all nodes in duplicate subtree
def sum_dups(dups):
    return sum([len(v) for k,v in dups.items()])

# Find the dublicates group containing the node with the shortest breadcrumb
def merge_identical_subtree_once(tree, cut_offs=[], also_leafs=False):
    dups = tree.find_duplicates_by_name()
    for k in sorted(dups, key=lambda k: min([len(n.breadcrumb())] for n in dups[k])): # sort by top most duplicate entry
        entries = sorted(dups[k], key= lambda n: len(n.breadcrumb())) # sort by len(breadcrumb)
        for a , b in combine_list(entries):            
            restriction = also_leafs or (a.is_intermediate_node() and b.is_intermediate_node()) # merge either all or only intermediate leaves
            if restriction and a.compare_tree(b) :
                cut_offs.append(('merge', b, b.breadcrumb(), a.breadcrumb())) # write to cut_offs
                #* print merged node subtree to cut_offs?
                b.disconnect_from_parent() #* merge dublicate subtrees
                return True
    return False

# runs check_and_merge as long as duplicates are detected
def merge_identical_subtrees(tree, *args, **kwargs):
    merged = 0
    while True:
        if not merge_identical_subtree_once(tree, *args, **kwargs):
            break
        merged +=1
    return merged
    
# write graph to file in the format of graphviz
def write_graph(tree, filename, depth):        
    with open(filename, 'w+') as f: #w+ creates file if it does not exist
        f.write("digraph {\n")
        f.write("{\n")
        tree.printGraphNode(f, depth)
        f.write("}\n")
        tree.printGraph(f, depth)
        f.write("}\n")
        f.close()

# write tree to file in the format of .json
def write_json(tree, filename):        
    with open(filename, 'w+') as f:
        f.write(str(tree))
        f.close()

# write statistics to file to be further process by gnuplot
def write_tsv(combined_dict, filename):
    keys = "\t".join(['Level','#Nodes','#Taggings'])
    with open(filename, 'w+') as f:
        f.write(keys+"\n")
        for k , v in combined_dict.items():
            #print(combined_dict[k])
            values = "\t".join(map(str, combined_dict[k]))
            f.write(str(k)+"\t"+values+"\n")
        f.close()

# write cut offs to file
## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)
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

# write cut offs to file
## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)
def write_cut_offs_trim(elem, f, *args, **kwargs):
    _, ch, p = elem
    def visitor(n):
        if n.data != -1:
            f.write(f'{p.root().name}:{p.root().name}:{p.name}:{n.name}\n')
    ch.visit(visitor)

# write cut offs to file
## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)    
#def write_cut_offs_merge():

# write cut offs to file
## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)
def write_cut_offs_split(elem, f, *args, **kwargs):
    _, ch, p = elem
    print('split: ', ch.name, " -> ", cut_off_str(p.breadcrumb()))
    
# write cut offs to file, verbose for debugging
def write_cut_offs_verbose_absorbe(elem, f, removed_nodes=None, *args, **kwargs):
    _, ch, breadcrumb, orig_bcstr = elem
    assert breadcrumb_str(breadcrumb) == orig_bcstr
    assert removed_nodes is not None
    final_bc = removed_nodes.resolve(breadcrumb)
    final_parent = final_bc[len(final_bc)-1]
    if final_parent.name == ch.name:
        final_parent = final_bc[len(final_bc)-2]
    final_root = final_parent.root()
    f.write(f'ABSORBE: {final_parent.breadcrumb_str()} : <- {ch.name} | data: {ch.data}\n')

# write cut offs to file, verbose for debugging
def write_cut_offs_verbose_trim(elem, f, *args, **kwargs):
    _, ch, p = elem
    def visitor(n):
        p_bc_str = p.breadcrumb_str()
        cutoff = n.breadcrumb_str()
        f.write(f'TRIM: {p_bc_str} : <- {cutoff} | data: {n.data}\n')
    ch.visit(visitor)

def write_cut_offs_verbose_merge(elem, f, *args, **kwargs):
    _, b, b_bc, a_bc = elem
    f.write(f'MERGE: \n{breadcrumb_str(a_bc)} : <- \n{breadcrumb_str(b_bc)}\n')
    
# handler type
write_cut_offs_types = {
    'absorbe': write_cut_offs_absorbe,
    'trim': write_cut_offs_trim,
    #'split': write_cut_offs_split,
    #'merge': write_cut_offs_merge,
}

# handler type verbose
write_cut_offs_verbose_types = {
    'absorbe': write_cut_offs_verbose_absorbe,
    'trim': write_cut_offs_verbose_trim,
    #'split': write_cut_offs_verbose_split,
    'merge': write_cut_offs_verbose_merge,
}

# initializes writing of cut offs to file
def write_cut_offs(cut_offs, filename, handlers=write_cut_offs_types, *args, **kwargs):
    with open(filename, 'w+') as f:
        for elem in cut_offs:
            elem_type = elem[0]
            handler = handlers.get(elem_type)
            if handler:
                handler(elem, f, *args, **kwargs)
        f.close()

# combines two dictionaries to one
def combine_dicts(dict1, dict2):
    dict_list = [dict1, dict2]            
    from collections import defaultdict
    #defaultdict does not require consistent keys across dictionaries, and  maintains O(n) time 
    d = defaultdict(list)
    for elem in dict_list:
        for k, v in elem.items():
            #print(k, v)
            d[k].append(v)
    return d

# find all combinations of two items in mylist
def combine_list(mylist): 
    l = len(mylist)
    i = 0
    ret = []
    for i in range(0, l-1):
        for j in range(i+1, l):
            ret.append( (mylist[i], mylist[j]) )
    return ret

# reads tag_count.csv (extracted from database) into a python dictionary
def read_object_tag_relation(filename):
    object_tag_relation = {}
    with open(filename) as file:
        file.readline() #trow away first line
        for line in file:
            (key, val) = line.split(",")
            object_tag_relation[int(key)] = int(val)
    return object_tag_relation

def print_trees(list_of_trees):
    print(f'# trees : {len(list_of_trees)}')
    for elem in list_of_trees:
        if len(elem.name) < 8:
            print('{}\t\t size: {}\t height: {}'.format(elem.name, elem.count_BFS(lambda n: True), elem.count_height()))
        else:
            print('{}\t size: {}\t height: {}'.format(elem.name, elem.count_BFS(lambda n: True), elem.count_height()))


# print statistics, and write tsvfile to be further processed by gnuplot
def print_stats(tree, tsvfile):
    print("total nodes:\t\t",tree.count_BFS(lambda n: True))
    print("leafs:\t\t\t", tree.count_leafs())
    print("intermediate:\t\t", tree.count_intermediate())
    print("collapseble:\t\t", tree.count_collapsable_singular_chain())
    print("height:\t\t\t", tree.count_height())
    print("avg leaf to root:\t", tree.avg_leaf_to_root())
    print()

    print("(level: #child)")
    print(tree.count_widths())
    print("(level: avg child)")
    print(tree.avg_child())
    
    print("\nDUPLICATES")
    dups = tree.find_duplicates_by_name()
    print(f"different dups: {len(dups)}\tsum of dups: {sum_dups(dups)}")
    print_top10_dup(dups)
    dups_data = tree.find_duplicates_by_data()
    print(f"different dups: {len(dups_data)}\tsum of dups: {sum_dups(dups_data)}")
    #print_top10_dup(dups_data)
    
    object_tag_relation = read_object_tag_relation("/home/ek/Documents/Thesis/SQL/tag_count.csv")
    no_present_tag , missing_tagging, total, tag_dict = tree.count_taggings(object_tag_relation)
    
    print()
    print("tag '-1':\t\t", no_present_tag)
    print("tagging not in DB:\t",missing_tagging)
    print("total found tags:\t", total)
    print()
    
    comb_dict = combine_dicts(tree.count_widths(), tag_dict)
    write_tsv(comb_dict, tsvfile)
        
#foo = root.clone().lookup("animal")
#root.lookup("animal")
#print_stats(foo)
#print(' -> '.join(list(map(lambda n: n.name, foo.breadcrumb()))), "\n")
cut_offs=[]

print("-- UNCOMPRESSED --")
print_stats(root, 'stats_initial.tsv')
#root.print_tree()

print("-- MERGE --")
cnt_merged = merge_identical_subtrees(root, cut_offs=cut_offs)
print("Merged count: ",cnt_merged)
print_stats(root, 'stats_after_merge.tsv')

collapse_all_single_children(root)
print("-- COMPRESS SINGULAR CHAIN--")
print_stats(root, 'stats_remove_singular_chain.tsv')
#root.print_tree()

removed_nodes= NodeHistory()
root.visit(collapse_nodes, budget=20, cut_offs=cut_offs, removed_nodes=removed_nodes)
print("-- COMPRESS COLLAPSE --")
print_stats(root, 'stats_after_collapse.tsv')
#root.print_tree()

cnt_merged = merge_identical_subtrees(root, cut_offs=cut_offs, also_leafs=True)
print("Merged count: ",cnt_merged)
print_stats(root, 'stats_after_merge_leafs.tsv')


#write_json(root, 'new_org_optimized_max20.json')
#write_graph(root, 'graph.dot', 2)  
        
subtree_list = split_automated(root, minNodes=100, maxNodes=500, subtrees=[])
#print_trees(subtree_list)
    
    #*subtree.trim(5, cut_offs=cut_offs)
    
    #subtree.lookup("vehicle")
    #subtree.lookup("wheeled_vehicle")
    #subtree.lookup("truck")
    
#print("-- AFTER COLLAPSE --")
#print("cut_offs: ", len(cut_offs), "\t removed_nodes: ", len(removed_nodes))
    
#if removed_nodes is not None:
#    print("-- AFTER SPLIT --")
#    print("cut_offs: ", len(cut_offs))

#print("-- AFTER TRIM --")
#print("cut_offs: ", len(cut_offs) )

#print("\n--", subtree_list[9].name, "--")
#subtree_list[9].print_tree()

#default
write_cut_offs(cut_offs, 'cut_off.txt', removed_nodes=removed_nodes)
#debug
write_cut_offs(cut_offs, 'cut_off_verbose.txt', handlers=write_cut_offs_verbose_types, removed_nodes=removed_nodes)
