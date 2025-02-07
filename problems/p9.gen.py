import networkx as nx
import json
import random

# circular ladder

def save_to_file(graph, fire_starts, num_teams, file_path):
    data = {
        "graph": nx.node_link_data(
            graph
        ),
        "fire_starts": fire_starts,
        "num_teams": num_teams,
    }
    with open(file_path, "w") as f:
        json.dump(data, f)

def main():
    G = nx.circular_ladder_graph(1000)

    F = [101, 104, 111, 128]
    num_teams = 4
    print("Problem generated. Writing to 'p9.json'")
    save_to_file(G, F, num_teams, "problems/p9.json")

if __name__ == "__main__":
    main()

