import math
import random
import numpy as np
import queue
from collections import namedtuple
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)

# ▼▼▼ CONFIG ▼▼▼ #

max_gen = 1000
population_size = 500
stations = 10
lines = 5
con_count = ncr(stations, 2)
input_nodes = con_count + 3 # 10C2, 1 current node, 1 current line, 1 bias
output_nodes = stations + 2 # 10 choice, 1 next line, 1 done
submap = np.random.rand(stations, 2)
base_inp = 0.1
base_ntp = 0.45
la_min = 0.3
la_max = 1.2
serve_mgh = np.ones(stations) # service potential
for i in range(len(serve_mgh)):
    serve_mgh[i] *= np.random.normal(0.75, 0.15)

# ▲▲▲ CONFIG ▲▲▲ #

Edge = namedtuple('Edge', ['vertex', 'weight'])

gin = 0

def sig(input):
    return (1 / (1 + math.exp(-4.9 * input)))

class Graph(object):
    def __init__(self, vertex_count):
        self.vertex_count = vertex_count
        self.adjacency_list = [[] for _ in range(vertex_count)]

    def add_edge(self, source, dest, weight):
        assert source < self.vertex_count
        assert dest < self.vertex_count
        self.adjacency_list[source].append(Edge(dest, weight))
        self.adjacency_list[dest].append(Edge(source, weight))

    def get_edge(self, vertex):
        for e in self.adjacency_list[vertex]:
            yield e

    def get_vertex(self):
        for v in range(self.vertex_count):
            yield v

class Creature:
    def __init__(self):
        self.fitness = 9999999
        self.species = None
        self.genome = None

    def newGenome(self):
        _gin = 0
        self.genome = []
        for i in range(output_nodes):
            for j in range(input_nodes):
                self.genome.append([j, i + input_nodes, random.uniform(-1/input_nodes, 1/input_nodes), True, _gin])
                _gin += 1

        self.genome = np.array(self.genome, dtype=float)

    def fire(self, input):
        nodes = np.unique(self.genome[:,:2].flatten()) # Array of every unique node
        nodes.fill(0)
        temp = np.copy(self.genome)

        for i in range(input_nodes):
            nodes[i] = input[i]

        while len(temp) != 0:
            for i in range(len(temp) - 1, -1, -1):
                remaining = np.unique(temp[:,1:2])

                if not temp[i][0] in remaining:
                    if temp[i][0] < input_nodes:
                        nodes[int(temp[i][1])] += nodes[int(temp[i][0])] * temp[i][2]
                    else:
                        nodes[int(temp[i][1])] += sig(nodes[int(temp[i][0])]) * temp[i][2]

                    temp = np.delete(temp, i, axis=0)

        _output = []
        for i in range(output_nodes):
            _output.append(sig(nodes[i + input_nodes]))

        _output = np.copy(_output)

        return _output

def dijkstra(graph, source, dest):
    q = queue.PriorityQueue()
    parents = []
    distances = []
    start_weight = float("inf")

    for i in graph.get_vertex():
        weight = start_weight
        if source == i:
            weight = 0
        distances.append(weight)
        parents.append(None)

    q.put(([0, source]))

    while not q.empty():
        v_tuple = q.get()
        v = v_tuple[1]

        for e in graph.get_edge(v):
            candidate_distance = distances[v] + e.weight
            if distances[e.vertex] > candidate_distance:
                distances[e.vertex] = candidate_distance
                parents[e.vertex] = v
                # primitive but effective negative cycle detection
                if candidate_distance < -1000:
                    raise Exception("Negative cycle detected")
                q.put(([distances[e.vertex], e.vertex]))

    return distances[dest]

def euclid(a, b):
    return (((a[0] - b[0])**2) + ((a[1] - b[1])**2))**.5

def eval(input):
    print(input)
    fit = 0

    # INP and NTP weighted by lambda. Sum Distances. Weight Sum by lambda. Average Weighted Sums.
    reformatted = []
    counter = 0
    connections = input[:con_count]
    for j in range(stations):
        for k in range(j + 1, stations):
            # [start, end, line]
            if connections[counter] != 0:
                reformatted.append([j, k, connections[counter] * 5])
    reformatted = np.array(reformatted)
    reformatted = reformatted[reformatted[:,2].argsort()]
    # reformatted = [[s1, s2, l], [s1, s2, l], [s1, s2, l]]

    line_stops = []
    line_count = len(np.unique(reformatted[:,2]))
    # For each line
    translator = {}
    new_graph_node = 0
    for j in range(line_count):
        line_stops.append(np.unique(np.array([i for i in reformatted if i[2] == j + 1])[:,:2].flatten()))
        for k in line_stops:
            if not ((k, j + 1) in translator): # if translator doesn't contain (station, line)
                translator[(k, j + 1)] = new_graph_node
                new_graph_node += 1

    for i in range(stations):
        λ = serve_mgh[i]
        G = Graph(stations**2)
        w_sum = 0

        # construct the chains
        for k in reformatted:
            station1 = translator[(k[0], k[2])]
            station2 = translator[(k[1], k[2])]
            G.add_edge(station1, station2, euclid(submap[k[0]], submap[k[1]]) + (base_inp * λ))

        # connect transfers
        for k in range(len(line_stops)):
            for j in range(k + 1, len(line_stops)):
                for l in line_stops[k]:
                    for m in line_stops[j]:
                        if (l == m):
                            G.add_edge(translator[(l, k + 1)], translator[(m, j + 1)], base_ntp * λ)

        for k in range(stations):
            if k != i:
                dtba = float('inf')
                for j in range(len(line_stops)):
                    if i in line_stops[j]:
                        for l in range(len(line_stops)):
                            if k in line_stops[l]:
                                dtba = min(dtba, dijkstra(G, translator[(i, j + 1)], translator[k, l + 1]))
                # ii = np.where(values == searchval)[0]
                w_sum += dtba
        fit += w_sum * λ

    return fit/stations

def main():
    pop = [Creature() for i in range(population_size)]
    for i in pop:
        i.newGenome()

    for g in range(max_gen):
        for i in pop:
            ip = np.zeros(input_nodes)
            ip[-2] = 0.2 # Line
            ip[-1] = 1 # Bias
            isolated = [i for i in range(stations)]

            output = i.fire(ip)
            prev_out = output
            choice = output[:stations].argsort()[-1:][0]
            ip[-3] = (choice + 1) / stations
            isolated.remove(choice)
            inline = [choice]


            done = False
            while not done:
                # Fire the Feed Forward Network
                output = i.fire(ip)
                print(output[:stations].argsort()[::-1])
                print(ip)

                # Order the index of the station with the highest probability
                goto = output[:stations].argsort()[::-1]
                for l in goto:
                    # if that station hasn't been visited yet
                    if not (l in inline):
                        # Connection = a tuple of current station and i
                        connection = (min((ip[-3] * stations) - 1, l), max((ip[-3] * stations) - 1, l))
                        _d = False
                        temp_ind = 0
                        # For each possible station combination
                        for j in range(stations):
                            for k in range(j + 1, stations):
                                # if that combination is the connection
                                if (j, k) == connection:
                                    _d = True
                                    # Add that connection to the current line
                                    if ip[temp_ind] == 0:
                                        ip[temp_ind] = ip[-2]
                                        # Set the new current node to the node that was just traveled to
                                        ip[-3] = (l + 1) / stations
                                        if l in isolated:
                                            isolated.remove(l)
                                        inline.append(i)
                                    break
                                temp_ind += 1
                            if _d == True:
                                break
                        break

                # if every station is in the line or next line, go to the next line
                if len(inline) == stations or output[stations:stations + 1] >= 0.5:
                    ip[-2] += 0.2
                    if ip[-2] > 1:
                        done = True

            eval(ip)

main()
