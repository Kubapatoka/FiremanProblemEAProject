import networkx as nx
import random
import numpy as np
import sys
import ProblemDef as pd

def generate_graph(n, p = 0.3, seed = 2137):
    phelp = (6*n-12)/(n*n-2*n)
    if p > phelp: p = phelp
    G = nx.gnp_random_graph(n, p, seed)
    return G


def fires_starting_points(n, k):
   return random.sample(range(n), k)

if __name__ == "__main__":
  args = len(sys.argv) - 1
  n = 100
  p = 0.3
  s = 2137

  if args > 0: n = int(sys.argv[1])
  if args > 1: p = float(sys.argv[2])
  if args > 2: s = int(sys.argv[3])
  
  G = generate_graph(n, p, s)

  k = 1
  if args > 3: k = int(sys.argv[4])
  
  F = fires_starting_points(n,k)

  counted = []
  max_num_of_man = 0
  for f in F:
    neighbors = G.neighbors(f)
    for h in neighbors:
        if h not in F and h not in counted:
           counted.append(h)
           max_num_of_man += 1
  
  num_of_firefighters = random.randint(1,max_num_of_man)
  prob = pd.FirefighterProblem(G,F,num_of_firefighters)
  prob.save_to_file("p2.json")
