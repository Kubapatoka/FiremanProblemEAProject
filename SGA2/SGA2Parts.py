from ProblemDef2 import IncrementalFirefighterProblem 
import random
from Utils import *
import statistics
import numpy as np
from Evaluators import *

# populationInitializer
def basicPI(populationSize : int, problem : IncrementalFirefighterProblem, evaluator):
    N = problem.num_teams
    chromosomeSize = problem.graph.number_of_nodes()

    newPopulation = []

    for i in range(populationSize):
        candidate = np.random.permutation(chromosomeSize)
        newPopulation.append(tuple([candidate, evaluator(candidate, problem)]))
    return newPopulation


# parentsSelector
def basicParentsSelector(population):
    pos1 = random.randint(0, len(population)-2)
    pos2 = random.randint(pos1, len(population)-1)

    return tuple([pos1, pos2])

def rankingParentsSelector(population):
    prob = random.randint(1,100)
    popSize = len(population)

    if  prob < 80:
        pos1 = random.randint(0, (popSize/10)-2)
    elif  prob < 95:
        pos1 = random.randint(popSize/10, (popSize/2)-2)
    else:
        pos1 = random.randint(popSize/2, popSize-2)

    pos2 = pos1
    while pos2 == pos1:
        prob = random.randint(1,100)
        if  prob < 80:
            pos2 = random.randint(0, (popSize/10)-1)
        elif  prob < 95:
            pos2 = random.randint(popSize/10, (popSize/2)-1)
        else:
            pos2 = random.randint(popSize/2, popSize-1)

    return tuple([pos1, pos2])


#PMX
def PMX(P1, P2):
    size = len(P1)
    start, end = sorted(random.sample(range(len(P1)), 2))
    changed = [0]*size
    iP1 = np.argsort(P1)
    iP2 = np.argsort(P2)
    
    for i in range(start, end+1):
        if changed[i] == 1:
            continue
        
        a, b = P1[i], P2[i]
        changed[i] = 1
        P1[i], P2[i] = P2[i], P1[i]
        j = iP2[a]
        while j>=start and j<=end and j!=i:
            a = P1[j]
            P1[j], P2[j] = P2[j], P1[j]
            changed[j] = 1
            j = iP2[a]
        if j == i:
            # case where whole cycle was enclised in selected chunk
            continue
            
        k = iP1[b]
        while k>=start and k<=end:
            b = P2[k]
            P1[k], P2[k] = P2[k], P1[k]
            changed[k] = 1
            k = iP1[b]
            
        P1[k], P2[j] = P2[j], P1[k]
    
    return P1, P2

# crossover
def PMXCrossover(p1 :list[int], p2 :list[int]):
    return PMX(p1,p2)

def OXCrossover(P1, P2):
    size = len(P1)
    a, b = sorted(random.sample(range(size), 2))
    O1, O2 = np.array([-1]*len(P1)), np.array([-1]*len(P2))
    O1[a:b], O2[a:b] = P1[a:b], P2[a:b]
    
    def fill(O, P):
        idx = 0
        for x in P:
            if idx == a:
                idx = b
            if x not in O:
                O[idx] = x
                idx+=1
    fill(O1, P2)
    fill(O2, P1)
    
    return O1, O2


def CXCrossover(P1, P2):
    size = len(P1)
    start = random.choice(range(size))
    iP2 = np.argsort(P2)
    
    a = P1[start]
    P1[start], P2[start] = P2[start], P1[start]
    i = iP2[a]
    while i!=start:
        a = P1[i]
        P1[i], P2[i] = P2[i], P1[i]
        i = iP2[a]
    
    return P1, P2

def PBXCrossover(P1, P2):
    size = len(P1)
    sample_size = int(random.triangular(1, size, size/3))
    inds = set(sorted(random.sample(range(size), sample_size)))
    O1 = np.array([x if i in inds else None for i, x in enumerate(P1)])
    O2 = np.array([x if i in inds else None for i, x in enumerate(P2)])
    
    def fill(O, P):
        idx = 0
        for x in P:
            while idx < len(O) and O[idx] is not None:
                idx+=1
            if x not in O:
                O[idx] = x
                idx+=1
    fill(O1, P2)
    fill(O2, P1)
    
    return O1, O2

def PBXCrossover(P1, P2):
    size = len(P1)
    sample_size = random.choice(range(size))
    inds = set(sorted(random.sample(range(size), sample_size)))
    O1 = np.array([x if i in inds else None for i, x in enumerate(P1)])
    O2 = np.array([x if i in inds else None for i, x in enumerate(P2)])
    
    idx1 = 0
    idx2 = 0
    for x, y in zip(P1, P2):
        while idx1 < len(O1) and O1[idx1] is not None:
            idx1+=1
        while idx2 < len(O2) and O2[idx2] is not None:
            idx2+=1
        if y not in O1:
            O1[idx1] = y
            idx1+=1
        if x not in O2:
            O2[idx2] = x
            idx2+=1
        
    return O1, O2

def OBXCrossover(P1, P2):
    size = len(P1)
    sample_size = int(random.triangular(1, size, size/3))
    elms = set(sorted(random.sample(range(size), sample_size)))
    O1 = np.array([x if x in elms else None for x in P1])
    O2 = np.array([x if x in elms else None for x in P2])
    
    def fill(O, P):
        idx = 0
        for x in P:
            while idx < len(O) and O[idx] is not None:
                idx+=1
            if x not in O:
                O[idx] = x
                idx+=1
    fill(O1, P2)
    fill(O2, P1)
    
    return O1, O2

# mutator
def noMutator(genotype :list[int]):
    return genotype

def basicMutator(genotype :list[int]):
    chromosomeSize = len(genotype)
    
    pos1 = random.randint(0, chromosomeSize-1)
    pos2 = random.randint(0, chromosomeSize-1)
    
    temp = genotype[pos1]
    genotype[pos1] = genotype[pos2]
    genotype[pos2] = temp

    return genotype


def reverse_sequence_mutation(p: list[bool]):
    a = np.random.choice(len(p), 2, False)
    i, j = a.min(), a.max()
    q = p.copy()
    q[i:j+1] = q[i:j+1][::-1]
    return q

inc = IncrementalMainEvaluator()
# evaluator
def basicEvaluator(genotype :list[bool], problem : IncrementalFirefighterProblem):
    return inc(problem, genotype)

cumm = IncrementalCummulativeEvaluator()
def CummulativeEvaluator(genotype :list[bool], problem : IncrementalFirefighterProblem):
    return cumm(problem, genotype)
