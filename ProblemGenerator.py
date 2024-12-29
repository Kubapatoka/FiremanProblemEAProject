import networkx as nx
import random
import numpy as np
import sys


def generate_graph(n, p = 0.3, seed = 2137):
    G = nx.gnp_random_graph(n, p, seed)
    return G


if __name__ == "__main__":
  args = len(sys.argv) - 1
  n = 10
  p = 0.3
  s = 2137

  if args > 0: n = int(sys.argv[1])
  if args > 1: p = float(sys.argv[2])
  if args > 2: s = int(sys.argv[3])
  
  print(generate_graph(n, p, s))