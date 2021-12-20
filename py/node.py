# return a string representation of path in tree
def breadcrumb_str(breadcrumb):
    path = ' -> '.join(list(map(lambda n: n.name, breadcrumb)))
    if path == "":
        path = 'root'
    return path
