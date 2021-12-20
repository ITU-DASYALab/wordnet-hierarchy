import json
import argparse
from os import posix_fallocate
from node import Node,NodeHistory,breadcrumb_str
from writeCutOffs import write_cut_offs, write_cut_offs_verbose

metadata = {}

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
def load_data_2(filename):
    global metadata
    metadata['input']={'filename':filename}
    with open(filename, 'r') as f:
    #with open('new_org.json', 'r') as f:
        tree = json.load(f)
        #print( from_json_to_tree(tree) )
    return from_json_to_tree(tree)

def write_metadata_to_json():
    with open('metadata.json', 'w+') as f:
        json.dump(metadata, f)
        f.close()

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
def split_automated(tree, minNodes=50, maxNodes=500, cut_offs=[], subtrees=[]):
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
    #print("(level: avg child)")
    #print(tree.avg_child())
    
    print("\nDUPLICATES")
    dups = tree.find_duplicates_by_name()
    print(f"different dups by name: {len(dups)}\tsum of dups: {sum_dups(dups)}")
    dups_data = tree.find_duplicates_by_data()
    print(f"different dups by data: {len(dups_data)}\tsum of dups: {sum_dups(dups_data)}")
    
    object_tag_relation = read_object_tag_relation("/home/ek/Documents/Thesis/SQL/tag_count.csv")
    no_present_tag , missing_tagging, total, tag_dict = tree.count_taggings(object_tag_relation)
    
    print()
    print("tag '-1':\t\t", no_present_tag)
    print("tagging not in DB:\t",missing_tagging)
    print("total found tags:\t", total)
    print()
    
    comb_dict = combine_dicts(tree.count_widths(), tag_dict)
    write_tsv(comb_dict, tsvfile)

def main():
    root = load_data_2('jsTree/tree_t5c0.json')
    cut_offs=[]

    print("-- UNCOMPRESSED --")
    print_stats(root, 'stats_initial.tsv')
    #root.print_tree()

    #print("-- MERGE --")
    #cnt_merged = merge_identical_subtrees(root, cut_offs=cut_offs)
    #print("Merged count: ",cnt_merged)
    #print_stats(root, 'stats_after_merge.tsv')

    collapse_all_single_children(root)
    print("-- COMPRESS SINGULAR CHAIN--")
    print_stats(root, 'stats_remove_singular_chain.tsv')
    #root.print_tree()

    removed_nodes= NodeHistory()
    root.visit(collapse_nodes, budget=20, cut_offs=cut_offs, removed_nodes=removed_nodes)
    print("-- COMPRESS COLLAPSE --")
    print_stats(root, 'stats_after_collapse.tsv')
    #root.print_tree()

    #TODO bygge script der tager data fra json og kører plots
    # jq -r .input.filename metadata.json

    #write_json(root, 'new_org_optimized_max20.json')
    #write_graph(root, 'graph.dot', 2)  
            
    subtree_list = split_automated(root, minNodes=100, maxNodes=500, cut_offs=cut_offs, subtrees=[])

    i = 0
    #for subtree in subtree_list:
    #    i += 1
    #    print("-- MERGED --", subtree.name)
    #    cnt_merged = merge_identical_subtrees(subtree, cut_offs=cut_offs, also_leafs=True)
    #    print("Merged count: ",cnt_merged)
    #    print_stats(subtree, f'stats_after_merge_sub{i}.tsv')
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
    write_cut_offs_verbose(cut_offs, 'cut_off_verbose.txt', removed_nodes=removed_nodes)

    write_metadata_to_json()

if __name__ == '__main__':
    main()
