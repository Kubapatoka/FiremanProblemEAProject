import networkx as nx
import copy
import json
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

@classmethod
def load_from_file(cls, file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    graph = nx.node_link_graph(data["graph"])  # Odtworzenie grafu z danych JSON
    fire_starts = data["fire_starts"]
    num_teams = data["num_teams"]
    return cls(graph, fire_starts, num_teams)
