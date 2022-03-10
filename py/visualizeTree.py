import json

#with open('jsTree/noTags.json', 'r') as f:
#with open('jsTree/siblings.json', 'r') as f:
#with open('jsTree/testTree.json', 'r') as f:
with open('jsTree/tree_t5c0.json', 'r') as f:
#with open('jsTree/tree_t5c3.json', 'r') as f:

#with open('jsTree/test.json', 'r') as f:

    tree = json.load(f)
    print(f)

#print(tree['name'])
#print(tree)

def recPrint(treeDict, bend):
    print(" "*bend +treeDict['name'])
    if 'children' in treeDict:
        for i in treeDict['children']:
            recPrint(i, bend+1)

recPrint(tree, 0)

def printSubTree(treeDict, bend, searchTerm, doPrint): #obs on doblicate subtrees
    #doPrint = False
    if doPrint:
        print(" "*bend +treeDict['name'])
    if 'children' in treeDict:
        for i in treeDict['children']:
            if treeDict['name'] == searchTerm:
                doPrint = True
            printSubTree(i, bend+1, searchTerm, doPrint)

#printSubTree(tree, 0, 'leukocyte', False)

def find_tag(treeDict, tag, level, options):
    #print(tag)
    #print(treeDict['name'])
    if not treeDict['name'] == tag:
        #print(options)
        more_options = options + len(treeDict['children'])
        #print(more_options)
        for i in treeDict['children']:
            find_tag(i, tag, level+1, more_options)
    else:
        print (level, options)

#find_tag(tree, "a.a", 0, 0)
#find_tag(tree, "a.b.a", 0, 0)

#def count_subtree(treeDict, level, options):
    

def rec_max_height(treeDict, level):
    #print(treeDict['name'])
    if 'children' in treeDict:
        maxH = 0
        for i in treeDict['children']:
            childH = rec_max_height(i, level+1)
            if childH > maxH:
                maxH = childH
        return maxH
    else:
        return level

print("Max tree height: " + str(rec_max_height(tree,0)))


def rec_how_many_children(treeDict):
    #print(treeDict['name'])
    if 'children' in treeDict:
        numChild = len(treeDict['children'])
        if numChild > 1:
            print(treeDict['name']+" "+str(numChild))
        for i in treeDict['children']:
            childW = rec_how_many_children(i)

#print('how many children')
#rec_how_many_children(tree)

def rec_only_child(treeDict): 
    if 'children' in treeDict:
        sum = 0
        if len(treeDict['children']) == 1:
            sum += 1
        for i in treeDict['children']:
            sum += rec_only_child(i)
        return sum
    else:
        return 0

#print("Nodes with one child: " + str(rec_only_child(tree)))

def histogram(treeDict, hist):
    if 'children' in treeDict:
        numChild = len(treeDict['children'])
        if not numChild in hist:
            hist[numChild] = 0
        hist[numChild] = hist[numChild] + 1
        for i in treeDict['children']:
            histogram(i, hist)

dt = {}
histogram(tree, dt)

#sortDt = {} 
#for i in sorted(dt):
#   sortDt[i]=dt[i]

sorted_dict = dict(sorted(dt.items()))

#print(sortDt)
#print("-----------------------")
print(sorted_dict)

#visualise
# gennemsnitlige børn baseret på histogram
