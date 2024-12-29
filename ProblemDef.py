import networkx as nx
import json

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

@classmethod
def load_from_file(cls, file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    graph = nx.node_link_graph(data["graph"])  # Odtworzenie grafu z danych JSON
    fire_starts = data["fire_starts"]
    num_teams = data["num_teams"]
    return cls(graph, fire_starts, num_teams)