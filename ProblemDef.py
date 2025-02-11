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

    def visualize_fire(
        self,
        displayer: Displayer,
        fireman,
        **kwargs,
    ):
        return displayer.simulate_fire(self.graph, self.fire_starts, fireman, **kwargs)

    def visualize_fires(
        self,
        displayer: Displayer,
        fireman_placements,
        **kwargs,
    ):
        return displayer.simulate_multiple_fireman_scenarios(
            self.graph, self.fire_starts, fireman_placements, **kwargs
        )

    def visualize_fire_without_burned(self, displayer: Displayer, fireman, **kwargs):
        return displayer.simulate_fire_lite(
            self.graph, self.fire_starts, fireman, **kwargs
        )

    def count_burned_verts(self, fireman:list[int]):
        number_of_nodes = self.graph.number_of_nodes()
        burned = np.repeat(False, number_of_nodes)

        #print(fireman)
        queue = []
        for f in self.fire_starts:
            queue.append(f)

        while queue:
            first_elem = queue.pop(0)
            burned[first_elem] = True
            #print("burn ", first_elem)
            neighbours = self.graph.neighbors(first_elem)
            for n in neighbours:
                if (n not in queue) and (not burned[n]) and (n not in fireman):
                    #print("add to queue ", n)
                    queue.append(n)

        burned_number = 0
        for i in range(number_of_nodes):
            if burned[i]:
                burned_number += 1
        burned_number -= len(self.fire_starts)

        return burned_number
    
    
    def count_burned_verts_and_rounds(self, fireman):
        number_of_nodes = self.graph.number_of_nodes()
        burned = np.repeat(False, number_of_nodes)
        #print("count_burned_verts ", number_of_nodes, " ", len(self.fire_starts))

        queue = []
        for f in self.fire_starts:
            queue.append((f,0))

        round_count = 0
        while queue:
            (first_elem,r) = queue.pop(0)
            round_count = max(round_count,r)
            if burned[first_elem]:continue
            burned[first_elem] = True
            #print("burn ", first_elem)
            neighbours = self.graph.neighbors(first_elem)
            for n in neighbours:
                if (n,r+1) in queue or burned[n] or n in fireman:
                    continue
                queue.append((n,r+1))

        burned_number = 0
        for i in range(number_of_nodes):
            if burned[i]:
                burned_number += 1
        burned_number -= len(self.fire_starts)

        return (burned_number,round_count)
    
    def count_burned_verts_and_fire_motion(self, fireman):
        number_of_nodes = self.graph.number_of_nodes()
        burned = np.repeat(False, number_of_nodes)

        fire_steps = []

        queue = []
        first_round = ([],0)
        for f in self.fire_starts:
            first_round[0].append(f)
        queue.append(first_round)

        round_count = 0
        while queue:
            (fire_line, r) = queue.pop(0)
            round_count = max(round_count,r)
            fire_steps.append(len(fire_line))
            new_fire_line = []

            for fire_node in fire_line:
                if burned[fire_node]:continue
                burned[fire_node] = True
                neighbours = self.graph.neighbors(fire_node)
                for n in neighbours:
                    if n in fire_line or burned[n] or n in fireman or n in new_fire_line:
                        continue
                    new_fire_line.append(n)
            if len(new_fire_line)>0 : queue.append((new_fire_line,r+1))

        burned_number = 0
        for i in range(number_of_nodes):
            if burned[i]:
                burned_number += 1
        burned_number -= len(self.fire_starts)

        return (burned_number,round_count, fire_steps)
    
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
