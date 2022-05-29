#!/bin/python3
#https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
# https://stackoverflow.com/questions/9735172/2d-array-path-finding

from collections import defaultdict
import random
import time
import pprint

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are NOT bi-directional
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight

def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # print(shortest_paths,len(shortest_paths))
    # Work back through destinations in shortest path
    path = []
    weight = shortest_paths[current_node][1]
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    print(weight)
    return path

def tupleNeighbourgh(i,j,H,W):
    l = []
    l.append([(i-1)%H,(j)%W])
    l.append([(i)%H,(j-1)%W])
    l.append([(i)%H,(j+1)%W])
    l.append([(i+1)%H,(j)%W])
    return l

random.seed(time.time())
W = 120
H = 60
map=[]
for i in range(H):
    map.append([])
    for j in range(W):
        map[-1].append(1)
        # map[-1].append(random.randint(0, 5))

startPath=str(random.randint(0, H-1))+'-'+str(random.randint(0, W-1))
endPath=str(random.randint(0, H-1))+'-'+str(random.randint(0, W-1))
# startPath='0-0'
# endPath='1-1'

c=[random.randint(0, H-1),random.randint(0, W-1)]
r=random.randint(min(W,H)//3,min(W,H)//2)
n=random.randint(1, 5)
for i in range(c[0]-r+1,c[0]+r):
    for j in range(c[1]-r+1,c[1]+r):
        map[i%H][j%W]=n
        
# for i in range(H):
#     for j in range(W):
#         print(map[i][j],end='')
#     print()

graph = Graph()

edges=[]
for i in range(H):
    for j in range(W):
        neig = tupleNeighbourgh(i,j,H,W)
        for e in neig:
            edges.append((
                str(i)+'-'+str(j),
                str(e[0])+'-'+str(e[1]),
                map[e[0]][e[1]]
            ))

for edge in edges:
    graph.add_edge(*edge)

# print(dijsktra(graph, '02', '44'))
print(startPath, endPath)
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(graph.edges)
# pp.pprint(graph.weights)
print()
t0=time.time()
path = dijsktra(graph, startPath, endPath)
print(time.time()-t0)
        
for i in range(H):
    for j in range(W):
        if str(i)+'-'+str(j) in path :
            print(' ',end='')
        else:
            print(map[i][j],end='')
    print()
# print(edges)