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
    G = nx.turan_graph(25,5)

    F = random.randint(0,5)
    num_teams = 10
    print("Problem generated. Writing to 'p12.json'")
    save_to_file(G, F, num_teams, "problems/p12.json")

if __name__ == "__main__":
    main()

