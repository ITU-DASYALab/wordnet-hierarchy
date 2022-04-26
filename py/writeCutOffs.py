from node import breadcrumb_str

# return a string representation of cut offs. Used for cutting tree from top
def cut_off_str(breadcrumb):
    bc = breadcrumb.copy()
    bc.reverse()
    path = ' <- '.join(list(map(lambda n: n.name, bc)))
    if path == "":
        path = 'root'
    return path

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
        f.write(f'{ch.name}->{final_parent.name}\n')
        #f.write(f'{final_root.name}:{final_root.name}:{final_parent.name}:{ch.name}\n')

# write cut offs to file
## Format: TagsetName:HierarchyName:ParrentTagName:ChildTag:ChildTag:ChildTag:(...)
def write_cut_offs_trim(elem, f, *args, **kwargs):
    _, ch, p = elem
    def visitor(n):
        if n.data != -1:
            f.write(f'{n.name}->{p.name}\n')
            #f.write(f'{p.root().name}:{p.root().name}:{p.name}:{n.name}\n')
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

#TODO write cutoffs and filter for final position.

# initializes writing of cut offs to file
def write_cut_offs(cut_offs, filename, handlers=write_cut_offs_types, *args, **kwargs):
    with open(filename, 'w+') as f:
        for elem in cut_offs:
            elem_type = elem[0]
            handler = handlers.get(elem_type)
            if handler:
                handler(elem, f, *args, **kwargs)
        f.close()
        
def write_cut_offs_verbose(cut_offs, filename, *args, **kwargs):
    write_cut_offs(cut_offs, filename, handlers=write_cut_offs_verbose_types, *args, **kwargs)
