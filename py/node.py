# return a string representation of path in tree
def breadcrumb_str(breadcrumb):
    path = ' -> '.join(list(map(lambda n: n.name, breadcrumb)))
    if path == "":
        path = 'root'
    return path

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

# encapsules the concept of a node in the tree
class Node(object):
    def __init__(self, name, parent):
        self.parent = parent # Single object
        self.children = []  # Array of objects
        self.name = name
        self.data = -1 # from distinct_features_t25.json

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
    
    def count_taggings(self, taggings):
        worklist = [ self ]
        missing_tagging = total = 0
        while len(worklist) > 0:
            n = worklist.pop(0)
            if n.name not in taggings:
                #continue
                missing_tagging += 1
            else: 
                #print("tag: ", n.name, "\t#taggings: ",object_tag_relation[n.name])
                total += taggings[n.name]
            for ch in n.children:
                worklist.append(ch)
        return missing_tagging, total
    
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
        return {k: v for k, v in dups.items() if len(v) > 1} #filtering dict

    def find_duplicates_by_name(self):
        return self.find_duplicates(lambda n: n.name)

    def find_duplicates_by_data(self):
        return self.find_duplicates(lambda n: n.data)

    def find_duplicates_by_name_with_inconsistent_data(self):
        dups = {}
        dups_by_name = self.find_duplicates_by_name()
        for name, entries in dups_by_name.items():
            if len(entries) == 1:
                continue
            values = {}
            for n in entries:
                values[n.data] = values.get(n.data, 0) + 1
            if len(values) > 1: #n.data is inconsistent
                dups[name] = entries
        return dups
    
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
    # obs ! node.id is not the same as tag_id in
    # def count_taggings_id(self, object_tag_relation):
    #     worklist = [ (self, 0) ]
    #     no_present_tag = missing_tagging = total = 0
    #     tags = {}
    #     while len(worklist) > 0:
    #         n, depth = worklist.pop(0)
    #         if n.data == -1:
    #             no_present_tag += 1
    #         elif n.data not in object_tag_relation:
    #             missing_tagging += 1
    #         else: 
    #             #print("tag: ", n.data, "\t#taggings: ",object_tag_relation[n.data])
    #             total += object_tag_relation[n.data]
    #             tags[depth] = tags.get(depth, 0) + object_tag_relation[n.data]
    #         for ch in n.children:
    #             worklist.append((ch, depth+1))
    #     return no_present_tag, missing_tagging, total, tags
    
    # count number of taggings based on name of tag
    def count_taggings_name(self, object_tag_relation):
        worklist = [ (self, 0) ]
        missing_tagging = total = 0
        tags = {}
        while len(worklist) > 0:
            n, depth = worklist.pop(0)
            if n.name not in object_tag_relation:
                missing_tagging += 1
            else: 
                #print("tag: ", n.name, "\t#taggings: ",object_tag_relation[n.name])
                total += object_tag_relation[n.name]
                tags[depth] = tags.get(depth, 0) + object_tag_relation[n.name]
            for ch in n.children:
                worklist.append((ch, depth+1))
        return missing_tagging, total, tags
    
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
    def split(self, target_depth, cut_offs):
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
    def trim(self, target_depth, cut_offs):
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
    def absorb(self, other, cut_offs, removed_nodes):
        #print(f'  --absorbe-- self: {self.name} other: {other.name}')
        if other.data != -1: # do a check if there data is in distict features
            cut_offs.append(('absorbe', other, other.breadcrumb(), other.breadcrumb_str()))
        for ch in other.children.copy(): # other.children changes
            self.steal_and_append_child(ch, cut_offs=cut_offs, removed_nodes=removed_nodes)
        other.parent = None #ready for garbage collection
        other.children = []

    # removes childs from current position in tree and append to new parent
    def steal_and_append_child(self, ch, cut_offs, removed_nodes):
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
    def collapse(self, cut_offs, removed_nodes): 
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
        
    # not scope of this project
    def commonWords(self, words=[]):
        # input: list of common words
        # exclude advance search terms
            # possible extend to domain specific scientific words
        return

    # to string
    def __str__(self):
        if self.is_tagged() and len(self.children) == 0:
            #actual leaf
            #p = '"name":"' + str(self.name) + '","id":' + str(self.data)
            p = '"name":"' + str(self.name) + '"'
            #return p
        elif self.is_tagged() and len(self.children) > 0:
            # node with images & and is also a parent
            #p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
            p = '"name":"{name}","children":[{children}]'.format(name=self.name, children=', '.join(map(str, self.children)))
        else:
            # part of hierachy, no images attached
            #p = '"name":"{name}","id":{data},"children":[{children}]'.format(name=self.name, data=self.data, children=', '.join(map(str, self.children)))
            p = '"name":"{name}","children":[{children}]'.format(name=self.name, children=', '.join(map(str, self.children)))
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
