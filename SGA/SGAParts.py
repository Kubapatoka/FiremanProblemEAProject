from ProblemDef import FirefighterProblem 
import random
from Utils import *

# populationInitializer
def basicPI(populationSize : int, chromosomeSize : int, problem : FirefighterProblem, evaluator):
    N = problem.num_teams

    newPopulation = []

    for i in range(populationSize):
        candidate = [False for _ in range(chromosomeSize)]
        for j in range(N):
            pos = random.randint(0, chromosomeSize-1)
            while pos in problem.fire_starts or candidate[pos]:
                pos = random.randint(0, chromosomeSize-1)
            candidate[pos] = True
        newPopulation.append(tuple([candidate, evaluator(candidate, problem)]))
    return newPopulation

def randomVertAndPathPI(populationSize : int, chromosomeSize : int, problem : FirefighterProblem, evaluator):
    N = problem.num_teams

    newPopulation = []

    numVerts = problem.graph.number_of_nodes()

    for i in range(populationSize):
        fireman = []

        while len(fireman) < N:
            start_vert = random.randint(0, numVerts-1)
            while start_vert in problem.fire_starts:
                start_vert = random.randint(0, numVerts-1)

            fireman.append(start_vert)
            while len(fireman) < N:
                neigh = list(filter(lambda x: x not in fireman and x not in problem.fire_starts, problem.graph.neighbors(fireman[-1])))
                if len(neigh) == 0: break
                fireman.append(neigh[random.randint(0, len(neigh)-1)])

        candidate = fenotypeToGenotype(fireman, chromosomeSize)
        newPopulation.append(tuple([candidate, evaluator(candidate, problem)]))
    return newPopulation


def randomVertAndDistrictPI(populationSize : int, chromosomeSize : int, problem : FirefighterProblem, evaluator):
    N = problem.num_teams

    newPopulation = []

    numVerts = problem.graph.number_of_nodes()

    for i in range(populationSize):
        fireman = []

        while len(fireman) < N:
            start_vert = random.randint(0, numVerts-1)
            while start_vert in problem.fire_starts:
                start_vert = random.randint(0, numVerts-1)

            fireman.append(start_vert)
            neigh = list(filter(lambda x: x not in fireman and x not in problem.fire_starts, problem.graph.neighbors(start_vert)))

            while len(fireman) < N:
                newFireman = neigh[random.randint(0, len(neigh)-1)]
                neigh.remove(newFireman)
                fireman.append(newFireman)
                neigh.extend(list(filter(lambda x: x not in fireman and x not in problem.fire_starts and x not in neigh, problem.graph.neighbors(newFireman))))
                if len(neigh) == 0: break

        candidate = fenotypeToGenotype(fireman, chromosomeSize)
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

# crossover
def basicCrossover(p1 :list[bool], p2 :list[bool], problem : FirefighterProblem):
    chromosomeSize = problem.graph.number_of_nodes()
    pos1 = random.randint(0, chromosomeSize-2)
    pos2 = random.randint(pos1, chromosomeSize-1)

    c1 = []
    c2 = []
    for i in range(0, pos1):
        c1.append(p1[i])
        c2.append(p2[i])

    for i in range(pos1, pos2):
        c1.append(p2[i])
        c2.append(p1[i])

    for i in range(pos2, chromosomeSize):
        c1.append(p1[i])
        c2.append(p2[i])

    return tuple([c1,c2])

# mutator
def noMutator(genotype :list[bool], problem : FirefighterProblem):
    return genotype

def basicMutator(genotype :list[bool], problem : FirefighterProblem):
    chromosomeSize = problem.graph.number_of_nodes()
    
    firemans = genotypeToFenotype(genotype)
    pos = random.randint(0, len(firemans)-1)
    f = firemans[pos]

    pos = random.randint(0, chromosomeSize-1)
    while pos in problem.fire_starts or pos in firemans:
        pos = random.randint(0, chromosomeSize-1)

    firemans.remove(f)
    firemans.append(pos)

    return fenotypeToGenotype(firemans, chromosomeSize)


def neighbourMutator(genotype : list[bool], problem : FirefighterProblem):
    chromosomeSize = problem.graph.number_of_nodes()

    firemans = genotypeToFenotype(genotype)

    pos = random.randint(0, len(firemans)-1)

    f = firemans[pos]

    neigh = list(problem.graph.neighbors(f))
    for n in neigh:
        if n in firemans or n in problem.fire_starts:
            neigh.remove(n)

    if len(neigh) == 0:
        pos = random.randint(0, chromosomeSize-1)
        while pos in problem.fire_starts or pos in firemans:
            pos = random.randint(0, chromosomeSize-1)

        firemans.remove(f)
        firemans.append(pos)

        return fenotypeToGenotype(firemans, chromosomeSize)

    pos = random.randint(0, len(neigh)-1)

    firemans.remove(f)
    firemans.append(neigh[pos])

    return fenotypeToGenotype(firemans, chromosomeSize)


# fixer
def basicFixer(genotype :list[bool], problem : FirefighterProblem):
    for f in problem.fire_starts:
        if genotype[f]:
            genotype[f] = False

    firemanCount = 0
    chromosomeSize = problem.graph.number_of_nodes()
    for i in range(chromosomeSize):
        if genotype[i]:
            firemanCount += 1

    for i in range(firemanCount-problem.num_teams):
        pos = random.randint(0, chromosomeSize-1)
        while pos in problem.fire_starts or not genotype[pos]:
            pos = random.randint(0, chromosomeSize-1)
        genotype[pos] = False

    for i in range(problem.num_teams-firemanCount):
        pos = random.randint(0, chromosomeSize-1)
        while pos in problem.fire_starts or genotype[pos]:
            pos = random.randint(0, chromosomeSize-1)
        genotype[pos] = True

    return genotype

# evaluator
def basicEvaluator(genotype :list[bool], problem : FirefighterProblem):
    return problem.count_burned_verts(genotypeToFenotype(genotype))

def EffectiveUselessEvaluator(genotype :list[bool], problem : FirefighterProblem):
    fireman = genotypeToFenotype(genotype)
    burned = problem.count_burned_verts(fireman)
    (effective_count, useless_count) = problem.effective_and_useless_firefighters_count(fireman)
    return 2*burned + 10* useless_count - effective_count


def RoundCountEvaluator(genotype :list[bool], problem : FirefighterProblem):
    fireman = genotypeToFenotype(genotype)
    (burned,rounds) = problem.count_burned_verts_and_rounds(fireman)
    return 4*burned - rounds


def intermediateEvaluator(genotype :list[bool], problem : FirefighterProblem):
    fireman = genotypeToFenotype(genotype)
    (burned,rounds) = problem.count_burned_verts_and_rounds(fireman)
    (effective_count, useless_count) = problem.effective_and_useless_firefighters_count(fireman)
    return 4*burned - rounds + 10* useless_count - effective_count


def fireStepsEvaluator(genotype :list[bool], problem : FirefighterProblem):
    fireman = genotypeToFenotype(genotype)
    (burned,round_count, fire_steps) = problem.count_burned_verts_and_fire_motion(fireman)
    (effective_count, useless_count) = problem.effective_and_useless_firefighters_count(fireman)
    
    #TODO sth with fire_steps
    return 4*burned + 10*useless_count