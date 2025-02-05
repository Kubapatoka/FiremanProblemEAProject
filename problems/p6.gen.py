import networkx as nx
import json
import random

# Bigger than p5

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
    # Define the edges layer by layer
    edges = [
        # Bottom layer
        (0, 1), (1, 2),
        # Connections between bottom and second layers
        (0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6),
        # Second layer
        (3, 4), (4, 5), (5, 6),
        # Connections between second and third layers
        (3, 7), (3, 8), (4, 8), (4, 9), (5, 9), (5, 10), (6, 10), (6, 11),
        # Third layer
        (7, 8), (8, 9), (9, 10), (10, 11),
        # Connections between third and fourth layers
        (7, 12), (9, 14), (11, 16), 
        # Fourth layer
        (12, 13), (13, 14), (14, 15), (15, 16),
        # Connections between fourth and fifth layers
        (12, 17), (13, 17), (13, 18), (14, 18), (14, 19), (15, 19), (15, 20), (16, 20),
        # Fifth layer
        (17, 18), (18, 19), (19, 20),
        # Connections between fifth and top layer
        (17, 21), (18, 21), (18, 22), (19, 22), (19, 23), (20, 23),
        # Top layer
        (21, 22), (22, 23),
    ]

    # Create the undirected graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    curr_n = G.number_of_nodes()
    total_n = 1300
    
    for i in range(curr_n, total_n):
        v = random.randint(0,i-1)
        while v in list([7,9,11,16,14,12]):
            v = random.randint(0,i-1)
        G.add_edge(i, v)

    F = [1]
    num_teams = 3
    print("Problem generated. Writing to 'p6.json'")
    save_to_file(G, F, num_teams, "problems/p6.json")

if __name__ == "__main__":
    main()

