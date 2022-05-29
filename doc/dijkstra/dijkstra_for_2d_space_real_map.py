#!/bin/python3
#https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
# https://stackoverflow.com/questions/9735172/2d-array-path-finding

from collections import defaultdict
import random
import time
import copy

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
    return (path,weight)

def tupleNeighbourgh(i,j,H,W):
    l = []
    l.append([(i-1)%H,(j)%W])
    l.append([(i)%H,(j-1)%W])
    l.append([(i)%H,(j+1)%W])
    l.append([(i+1)%H,(j)%W])
    return l

random.seed(time.time())
W = 180
H = 60
map=[]
for i in range(H):
    map.append([])
    for j in range(W):
        map[-1].append(2)
        # map[-1].append(random.randint(0, 5))

startPath=[random.randint(0, H-1),random.randint(0, W-1)]
endPath=[random.randint(0, H-1),random.randint(0, W-1)]
# startPath=str(random.randint(0, H-1))+'-'+str(random.randint(0, W-1))
# endPath=str(random.randint(0, H-1))+'-'+str(random.randint(0, W-1))

r=4
for i in range(startPath[0]-r,startPath[0]+r+1):
    for j in range(startPath[1]-r,startPath[1]+r+1):
        if ( i - startPath[0] ) ** 2 + ( j - startPath[1] ) ** 2 <= r ** 2 :
            map[i%H][j%W]=1

for _ in range(4):
    c=[random.randint(0, H-1),random.randint(0, W-1)]
    r=random.randint(3,6)
    n=random.randint(0,1) * 3
    for i in range(c[0]-r,c[0]+r+1):
        for j in range(c[1]-r,c[1]+r+1):
            if ( i - c[0] ) ** 2 + ( j - c[1] ) ** 2 <= r ** 2 :
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

print(startPath, endPath)
sp = str(startPath[0])+'-'+str(startPath[1])
ep = str(endPath[0])+'-'+str(endPath[1])
t0=time.time()
(path,weight) = dijsktra(graph, sp, ep)
print(time.time()-t0)
print(weight)

for i in range(H):
    for j in range(W):
        if str(i)+'-'+str(j) in path :
            print(' ',end='')
        else:
            print(map[i][j],end='')
    print()
    
map2=[]
for i in range(H):
    map2.append([])
    for j in range(W):
        map2[-1].append('X')

t0=time.time()
d = 10
r=2
g=0
while True :
    p=8*(r-1)
    for i in range(p):
        if i // (p//4) == 0 :
            endPath = [ startPath[0] - r + 1 , startPath[1] - r + 1 + i%(p//4) ]
        elif i // (p//4) == 1 :
            endPath = [ startPath[0] - r + 1 + i%(p//4) , startPath[1] + r - 1 ]
        elif i // (p//4) == 2 :
            endPath = [ startPath[0] + r - 1 , startPath[1] + r - 1 - (i%(p//4)) ]
        elif i // (p//4) == 3 :
            endPath = [ startPath[0] + r - 1 -(i%(p//4)) , startPath[1] - r + 1 ]
        if r==2:
            print(endPath)
        ep = str(endPath[0])+'-'+str(endPath[1])
        res = dijsktra(graph, sp, ep)
        # print(p,i)
        # print(res)
        # print(sp,ep)
        if d <= int(res[1]) :
            g+=1
        else:
            map2[endPath[0]][endPath[1]] = str(map[endPath[0]][endPath[1]])
    if g == p :
        break
    r+=1
    g=0
print(time.time()-t0)

for i in range(H):
    for j in range(W):
        print(map2[i][j],end='')
    print()