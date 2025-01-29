from ProblemDef import FirefighterProblem 
import random

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

    # TODO: random vert and some path

# parentsSelector
def basicParentsSelector(population):
    pos1 = random.randint(0, len(population)-2)
    pos2 = random.randint(pos1, len(population)-1)

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

    pos = random.randint(0, chromosomeSize-1)
    while pos in problem.fire_starts or not genotype[pos]:
        pos = random.randint(0, chromosomeSize-1)
    genotype[pos] = False

    pos = random.randint(0, chromosomeSize-1)
    while pos in problem.fire_starts or genotype[pos]:
        pos = random.randint(0, chromosomeSize-1)
    genotype[pos] = True

    return genotype

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
    fireman = []
    for i in  range(problem.graph.number_of_nodes()):
        if genotype[i]:
            fireman.append(i)
    return problem.count_burned_verts(fireman=fireman)