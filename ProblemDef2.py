import networkx as nx
import copy
import json
import numpy as np
from Displayer import Displayer

class IncrementalFirefighterProblem:
    def __init__(self, graph: nx.Graph, fire_starts: list, num_teams_start: int, num_teams_increment: int):
        self.graph = graph
        self.fire_starts = fire_starts
        self.num_teams_start = num_teams_start
        self.num_teams_increment = num_teams_increment

    def save_to_file(self, file_path):
        data = {
            "graph": nx.node_link_data(self.graph),  # Konwersja grafu do formatu JSON-serializable
            "fire_starts": self.fire_starts,
            "num_teams_starts": self.num_teams_start,
            "num_teams_increments": self.num_teams_increment,
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

    def visualize_fire(
        self,
        displayer: Displayer,
        fireman,
        **kwargs,
    ):
        assert False, "Not implemented yet"
        return displayer.simulate_fire(self.graph, self.fire_starts, fireman, **kwargs)

    def visualize_fires(
        self,
        displayer: Displayer,
        fireman_placements,
        **kwargs,
    ):
        assert False, "Not implemented yet"
        return displayer.simulate_multiple_fireman_scenarios(
            self.graph, self.fire_starts, fireman_placements, **kwargs
        )

    def visualize_fire_without_burned(self, displayer: Displayer, fireman, **kwargs):
        assert False, "Not implemented yet"
        return displayer.simulate_fire_lite(
            self.graph, self.fire_starts, fireman, **kwargs
        )

    def eval(self, fireman):
        assert False, "Functionality moved to Evaluators.py"

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data["graph"])  # Odtworzenie grafu z danych JSON
        fire_starts = data["fire_starts"]
        num_teams = data["num_teams"]
        return cls(graph, fire_starts, num_teams)
