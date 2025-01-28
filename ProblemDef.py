import networkx as nx
import copy
import json
import numpy as np
from Displayer import Displayer

class FirefighterProblem:
    def __init__(self, graph: nx.Graph, fire_starts: list, num_teams: int):
        self.graph = graph
        self.fire_starts = fire_starts
        self.num_teams = num_teams

    def save_to_file(self, file_path):
        data = {
            "graph": nx.node_link_data(self.graph),  # Konwersja grafu do formatu JSON-serializable
            "fire_starts": self.fire_starts,
            "num_teams": self.num_teams
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

    def visualize_fire(self, displayer: Displayer, fireman, gif_path="fire_simulation.gif", fps=1):
        # Deep copy the graph to avoid modifying the original
        graph_copy = copy.deepcopy(self.graph)

        # Initialize attributes
        for node in graph_copy.nodes:
          graph_copy.nodes[node]["guarded"] = node in fireman
          graph_copy.nodes[node]["burned"] = False
          graph_copy.nodes[node]["on_fire"] = node in self.fire_starts

        # Run the simulation
        displayer.simulate_fire(graph_copy, gif_path, fps)

    def count_burned_verts(self, fireman):
        number_of_nodes = self.graph.number_of_nodes()
        burned = np.repeat(False, number_of_nodes)

        queue = []
        for f in self.fire_starts:
            queue.append(f)
            burned[f] = True

        while not queue:
            first_elem = queue.pop(0)
            neighbours = self.graph.neighbors(first_elem)
            for n in neighbours:
                if n in queue or burned[n] or n in fireman:
                    continue
                queue.append(n)

        burned_number = 0
        for i in range(number_of_nodes):
            if burned[i]:
                burned_number += 1
        burned_number -= len(self.fire_starts)

        return burned_number
    
    def effective_and_useless_firefighters_count(self, fireman):
        number_of_nodes = self.graph.number_of_nodes()
        burned = np.repeat(False, number_of_nodes)

        queue = []
        for f in self.fire_starts:
            queue.append(f)
            burned[f] = True

        while not queue:
            first_elem = queue.pop(0)
            neighbours = self.graph.neighbors(first_elem)
            for n in neighbours:
                if n in queue or burned[n] or n in fireman:
                    continue
                queue.append(n)

        effective_count = 0
        useless_count = 0

        for f in fireman:
            burn = False
            not_burn = False

            for n in self.graph.neighbors(f):
                if n in fireman: continue
                if burned[n]: burn = True
                else: not_burn = True
            
            if burn and not_burn: effective_count +=1
            if not_burn and not burn: useless_count += 1
        return (effective_count, useless_count)



    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data["graph"])  # Odtworzenie grafu z danych JSON
        fire_starts = data["fire_starts"]
        num_teams = data["num_teams"]
        return cls(graph, fire_starts, num_teams)
