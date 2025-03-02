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
    G = nx.barbell_graph(500, 100)

    F = [random.randint(0,500)]
    num_teams = 1
    print("Problem generated. Writing to 'p11.json'")
    save_to_file(G, F, num_teams, "problems/p11.json")

if __name__ == "__main__":
    main()

