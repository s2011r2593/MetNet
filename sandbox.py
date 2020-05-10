from collections import defaultdict
import random

random.seed(420)

dmat = []
temp = []
for i in range(5):
    for j in range(5):
        temp.append(0)
    dmat.append(temp)
    temp = []
for i in range(5):
    for j in range(5):
        if i != j and random.uniform(0,1) > 0.5:
            temp = random.uniform(1, 10)
            dmat[i][j] = temp
            dmat[j][i] = temp

connections = []
for i in range(len(dmat)):
    for j in range(len(dmat)):
        if i < j and dmat[i][j] != 0.0:
            connections.append([i, j])

class Graph(object):
    def __init__(self, connections):
        self._graph = defaultdict(set)
        self.add_connections(connections)

    def add_connections(self, connections):
        for n1, n2 in connections:
            self.add(n1, n2)

    def add(self, n1, n2):
        self._graph[n1].add(n2)
        self._graph[n2].add(n1)

    def delete(self, n1, n2):
        self._graph[n1].remove(n2)
        self._graph[n2].remove(n1)

    def find_path(self, n1, n2, path=[]):
        path = path + [n1]
        if n1 == n2:
            return path
        if n1 not in self._graph:
            return None
        for node in self._graph[n1]:
            if node not in path:
                new_path = self.find_path(node, n2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

g = Graph(connections)
print(g._graph[0])
