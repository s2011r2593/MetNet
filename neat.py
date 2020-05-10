from collections import defaultdict
import numpy as np
import pygame
import random
import math
import csv
from tqdm import tqdm

width = 1000
height = 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Neat Metro Net')
background_color = (252, 252, 252)

map_name = pygame.image.load('/Users/student/desktop/metnet/title.png')
mapw, maph = map_name.get_size()
map_name = pygame.transform.smoothscale(map_name, (int(mapw * 0.6), int(maph * 0.6)))

c = []
with open('/Users/student/desktop/metnet/a.csv', 'r') as f:
    reader = csv.reader(f)
    s_c = list(reader)
    for i in s_c:
        c.append([float(i[0]), float(i[1]), (42, 42, 42)])

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

def draw():
    screen.fill(background_color)
    screen.blit(map_name, (0, 10))
    for i in c:
        pygame.draw.circle(screen, i[2], (int(i[0]), int(i[1])), 2)
    pygame.display.update()

def dijkstra(n, start, d):
    unvisited = {node: None for node in n}
    visited = {}
    current = start
    currentDistance = 0
    unvisited[current] = currentDistance
    while True:
        temp = None
        for i in unvisited:
            if i != start and unvisited[i] == 0.0:
                temp = i
        if temp != None:
            del unvisited[temp]
        for neighbor, distance in d[current]:
            if neighbor not in unvisited: continue
            newDistance = currentDistance + distance
            if unvisited[neighbor] is None or unvisited[neighbor] > newDistance:
                unvisited[neighbor] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]

    return visited.items()

distmat = [[0 for i in range(len(c))] for j in range(len(c))]

for i in range(len(distmat)):
    for j in range(len(distmat)):
        distmat[i][j] = [j, (((c[i][0] - c[j][0])**2) + ((c[i][1] - c[j][1])**2))**(.5)]
        distmat[j][i] = [i, (((c[i][0] - c[j][0])**2) + ((c[i][1] - c[j][1])**2))**(.5)]

connections = []
for i in range(len(distmat)):
    for j in range(len(distmat)):
        if i < j and distmat[i][j] != 0:
            connections.append([i, j])

g = Graph(connections)

distances = []
for i in distmat:
    distances.append([j for j in i if j != 0.0])

all_dist = []

for i in tqdm(range(434)):
    all_dist.append(dijkstra(g._graph[i], i, distances))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    draw();
