from ProblemDef import FirefighterProblem
import itertools


def BruteForce(problem: FirefighterProblem):
    t = problem.num_teams
    #print("\nBruteForce for  ", t)

    n = problem.graph.number_of_nodes()

    best_c = []
    best_n = 1e9

    for comb in itertools.combinations([i for i in range(n) ], t):
        bn = problem.count_burned_verts(comb)
        #print(comb, " burned verts: ", bn)
        if bn == best_n:
            best_c.append(comb)
        if bn < best_n:
            best_n = bn
            best_c = []
            best_c.append(comb)
    #print(best_c, " burned verts: ", best_n)
    return tuple([best_c, best_n])