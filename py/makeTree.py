import json
import argparse
from os import posix_fallocate, remove, path
from node import Node,NodeHistory,breadcrumb_str
from writeCutOffs import write_cut_offs, write_cut_offs_verbose

ids = None 

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
def load_data_2(filename):
    with open(filename, 'r') as f:
    #with open('new_org.json', 'r') as f:
        tree = json.load(f)
        #print( from_json_to_tree(tree) )
    return from_json_to_tree(tree)

# merge all grandchildren at once. If all gr_ch doesn't fit in the max_children we don't merge.
def collapse_nodes(n, max_children, cut_offs=[], removed_nodes=None):
    #print('> ', n.name, max_children)
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
        if sum <= max_children:
            worklist = n.children.copy() # n.children is being modified during while loop
            for ch in worklist:
                if ch.is_intermediate_node():
                    ch.collapse(cut_offs=cut_offs, removed_nodes=removed_nodes)
                    rerun = True
                # else is_leaf() -> do nothing

# collapse all nodes that is a one_child and has one_child
def collapse_all_single_children(n, cut_offs, removed_nodes):
    def visitor(n):
        if n.should_collapse_singular_chain():
            n.collapse(cut_offs=cut_offs, removed_nodes=removed_nodes)
    i = 0 
    while n.count_collapsable_singular_chain() > 0:
        i += 1
        n.visit(visitor)
        assert i < 100 #safety feature to avoid eternal loop is non solvable due to a bug..

#def split_once():
    
# split tree from top
def split_and_eval_size(tree, cut_offs, minNodes=50, maxNodes=500, subtrees=[]):
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
def merge_identical_subtree_once(tree, cut_offs, also_leafs=False):
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
        
# write tree to file in the format of .json
def write_file(filename, data):        
    with open(filename, 'w+') as f:
        f.write(str(data))
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
def print_stats(tree, description, tsvfile, verbose=False):
    def vprint(*args):
        if verbose:
            print(*args)
            
    stats = {
        'description' : description,
    }
    vprint("--", description, "--")
    
    total_nodes = tree.count_BFS(lambda n: True)
    stats['total_nodes'] = total_nodes
    vprint("total nodes:\t\t",total_nodes)
    
    total_leafs = tree.count_leafs()
    stats['total_leafs'] = total_leafs
    vprint("leafs:\t\t\t", total_leafs)
    
    intermediate_nodes = tree.count_intermediate()
    stats['intermediate_nodes'] = intermediate_nodes
    vprint("intermediate:\t\t", intermediate_nodes)
    
    singular_chain_nodes = tree.count_collapsable_singular_chain()
    stats['singular_chain_nodes'] = singular_chain_nodes
    vprint("part of singular chain:\t", singular_chain_nodes)
    
    tree_height = tree.count_height()
    stats['tree_height'] = tree_height
    vprint("tree height:\t\t", tree_height)
    
    avg_leaf_to_root = tree.avg_leaf_to_root()
    stats[avg_leaf_to_root] = avg_leaf_to_root
    vprint("avg leaf to root:\t", tree.avg_leaf_to_root())

    tree_widths = tree.count_widths()
    stats['tree_widths'] = tree_widths
    vprint("(level: #child)\n", tree_widths)
    #print("(level: avg child)")
    #print(tree.avg_child())
    
    vprint("\nDUPLICATES")
    dups = tree.find_duplicates_by_name()
    stats['dups_name']=dups
    vprint(f"different dups by name:\t\t{len(dups)}\tsum of dups: {sum_dups(dups)}")
    dups_data = tree.find_duplicates_by_data()
    stats['dups_data']=dups_data
    vprint(f"different dups by data:\t\t{len(dups_data)}\tsum of dups: {sum_dups(dups_data)}")
    dups_name_data = tree.find_duplicates_by_name_with_inconsistent_data()
    stats['dups_name_inconsisten_data']=dups_name_data
    vprint(f"different dups by data&name:\t{len(dups_name_data)}\tsum of dups: {sum_dups(dups_name_data)}\n")
    
    object_tag_relation = read_object_tag_relation("/home/ek/Documents/Thesis/SQL/tag_count.csv")
    no_present_tag , missing_tagging, total_taggings, tag_dict = tree.count_taggings(object_tag_relation)
    stats['total_taggings']=total_taggings
    vprint(f"total taggings:\t{total_taggings}")
    stats['not_present_tags']=no_present_tag
    stats['missing_taggings']=missing_tagging
    vprint(f"tag '-1':\t{no_present_tag}\t tagging not in DB:\t{missing_tagging}\n")
    
    comb_dict = combine_dicts(tree_widths, tag_dict)
    write_tsv(comb_dict, tsvfile)
    
    return stats

#Class to contain state during different steps of optimization
class Compression_run:
    def __init__(self, filename, max_children, verbose=False):
        self._cut_offs = []
        self._removed_nodes = NodeHistory()
        self._max_children = max_children
        self._verbose = verbose
        self._subtree_ids_map = None
        self._subtrees = None
        self._root = load_data_2(filename)        
        self._metadata = {
            'input' : {'filename':path.basename(filename)},
            'outputs' : {},
            'subtrees' : {},
            'stats' : [],
        }
        basename = path.splitext(filename)[0]
        self._basename = f'{basename}_{self.get_postfix()}'
        
    def get_postfix(self): # TODO incl all relevant params
        return f'max{self._max_children}'
        
    def print_stats(self, description, filename, tree=None, is_subtree=False):
        if tree is None:
            tree = self._root
        filename = f'{self._basename}_{filename}.tsv'
        #TODO forskillige niveauer i stats & debug output + omstrukturering
        print("cut_offs: ", len(self._cut_offs), "\t removed_nodes: ", len(self._removed_nodes)) #relevant to debug collaps
        stats = print_stats(tree, description, filename, verbose=self._verbose)
        if not is_subtree:
            self._metadata['stats'].append(stats)
        stats['filename'] = path.basename(filename)
        return stats
    
    # removes nodes and their subtrees
    def clean_tree(self, excludes):
        def visitor(n):
            if n.name in excludes:
                n.disconnect_from_parent()
        self._root.visit(visitor)
        
    def clean_tree_with_file(self, exclude_file):
        if exclude_file != '':
            with open(exclude_file,'r') as f:
                excludes = set([s.strip() for s in f.readlines()])
                self.clean_tree(excludes)
                return True
        return False
    
    def collapse_all_single_children(self):
        collapse_all_single_children(self._root, self._cut_offs, self._removed_nodes)
        
    def approach_max_child_number(self):
        self._root.visit(collapse_nodes, self._max_children, cut_offs=self._cut_offs, removed_nodes=self._removed_nodes)

    def merge_identical_subtrees(self, tree=None, *args, **kwargs):
        if tree is None:
            tree = self._root
        return merge_identical_subtrees(self._root, cut_offs=self._cut_offs, *args, **kwargs)
    
    def split_into_subtrees(self, minNodes, maxNodes):
        assert self._subtrees is None
        subtrees = split_and_eval_size(self._root, minNodes=minNodes, maxNodes=maxNodes, cut_offs=self._cut_offs, subtrees=[])
        subtree_ids = {}
        i = 0
        for tree in subtrees:
            subtree_ids[tree] = i
            i += 1
        self._subtree_ids_map = subtree_ids
        self._subtrees = subtrees
        return subtrees
    
    def get_subtree_id(self, tree):
        return self._subtree_ids_map.get(tree)
        
    def write_subtree_stats(self, tree):
        id = self.get_subtree_id(tree)
        filename = f'subtree{id}'
        return self.print_stats(description=f'subtree{id}', filename=filename, tree=tree, is_subtree=True)
    
    def trim(self, depth, tree=None):
        if tree is None:
            tree = self._root
        tree.trim(depth, self._cut_offs)
    
    def store_output(self, name, data):
        if 'filename' in data:
            data['filename']=path.basename(data['filename'])
        outputs = self._metadata['outputs']
        outputs[name] = data
    
    def write_subtrees_json(self):
        data = ','.join([str(subtree) for subtree in self._subtrees])
        data = f'[{data}]'
        filename = f'{self._basename}_subtrees.json'
        write_file(filename, data)
        self.store_output('subtrees_json', {'filename' : filename})
        
    def write_subtrees(self):
        for subtree in self._subtrees:
            stats = self.write_subtree_stats(subtree)
            id = self.get_subtree_id(subtree)
            self._metadata['subtrees'][id] = stats
        #print_trees(self._subtrees)
        self.write_subtrees_json()
        
    def write_cut_offs(self, filename):
        filename = f'{self._basename}_{filename}'
        write_cut_offs(self._cut_offs, filename, removed_nodes=self._removed_nodes)
        self.store_output('write_cut_offs', {'filename' : filename})

    def write_cut_offs_verbose(self, filename):
        filename = f'{self._basename}_{filename}'
        write_cut_offs_verbose(self._cut_offs, filename, removed_nodes=self._removed_nodes)
        self.store_output('write_cut_offs_verbose', {'filename' : filename})

    def write_metadata_to_json(self, filename=None):   
        if filename is None:
            filename = f'{self._basename}_metadata.json'     
        with open(filename, 'w+') as f:
            json.dump(self._metadata, f)
            f.close()
        
    def write_json(self, filename=None, tree=None):
        if filename is None:
            filename = f'{self._basename}_out.json'
        if tree is None:
            tree = self._root
        write_file(filename, tree)
        if tree == self._root:
            self.store_output('final_json', {'filename' : filename})

def main():
    global ids
    
    HELP_HIEARCHY = 'WordNet Hiearchy File'
    HELP_ID_F = 'Distinct feature id file'
    HELP_EXCLUDE = 'File containing words to remove. 1 word per line'
    HELP_MAX_CHILDREN = 'maximum children per node'
    HELP_VERBOSE = 'enable verbose output'
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy_f', type=str, default='lsc_wn_hierarchy_t6.json')
    parser.add_argument('--id_f', type=str, default='distinct_features_t6.json')
    parser.add_argument('--exclude_f', type=str, default='')
    parser.add_argument('--max_children', type=int, default=20)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    with open(args.id_f,'r') as f: 
        ids = json.load(f)
                    
    run = Compression_run('jsTree/tree_t5c0.json', max_children=args.max_children, verbose = args.verbose)
    
    run.print_stats("UNCOMPRESSED", 'stats_initial')
    #root.print_tree()

    if run.clean_tree_with_file(args.exclude_f):
        run.print_stats("POST EXCLUDE", 'stats_post_exclude')


    run.collapse_all_single_children()
    run.print_stats("COMPRESS SINGULAR CHAIN", 'stats_remove_singular_chain')
    #root.print_tree()

    run.approach_max_child_number()
    run.print_stats("UTILIZE SIZE OF CHILDREN", 'stats_post_collapse')
    #root.print_tree()

    #* merge is last step
    #cnt_merged = run.merge_identical_subtrees()
    #print("Merged count: ",cnt_merged)
    #run.print_stats("MERGE", 'stats_after_merge')
    
    #TODO bygge script der tager data fra json og kører plots
    # jq . metadata.json

    run.write_json()
    #write_graph(root, 'graph.dot', 2)  
            
    subtree_list = run.split_into_subtrees(minNodes=100, maxNodes=500)        
    for subtree in subtree_list:
        cnt_merged = run.merge_identical_subtrees(subtree, also_leafs=True)
        #print("-- MERGED SUBTREE --", subtree.name)
        #print("Merged count: ",cnt_merged)
        run.trim(depth=5, tree=subtree)

    run.write_subtrees()

    #default
    run.write_cut_offs('cut_off.txt')
    #debug
    run.write_cut_offs_verbose('cut_off_verbose.txt')

    run.write_metadata_to_json()

if __name__ == '__main__':
    main()
