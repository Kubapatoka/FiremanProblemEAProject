import networkx as nx
import json
import random

# balanced tree 

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
    G = nx.balanced_tree(4,6)

    F = [12]
    num_teams = 1
    print("Problem generated. Writing to 'p10.json'")
    save_to_file(G, F, num_teams, "problems/p10.json")

if __name__ == "__main__":
    main()

