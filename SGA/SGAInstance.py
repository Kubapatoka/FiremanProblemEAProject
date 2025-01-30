from ProblemDef import FirefighterProblem 
import numpy as np


class SGAInstance:
    def __init__(self, file_name, populationInitializer, populationSize, generationsNumber, parentsSelector, crossover, mutator, fixer, mutationProb, evaluator):
        self.problem: FirefighterProblem = FirefighterProblem.load_from_file(file_name)
        self._populationInitializer = populationInitializer
        self._populationSize = populationSize
        self._chromosomeLength = self.problem.graph.number_of_nodes()
        #print("Init SGA instance. Chromosome length = {}", self._chromosomeLength)
        self._currentIteration = 0
        self._generationsNumber = generationsNumber
        self._parentsSelector = parentsSelector
        self._crossover = crossover
        self._mutationProb = mutationProb
        self._mutator   = mutator
        self._fixer   = fixer
        self.best_solution = None
        self._evaluator = evaluator

    def Mutation(self, candidates: list[list[bool]]):
        rv = []
        for c in candidates:
            if np.random.random() < self._mutationProb:
                rv.append(self._fixer(self._mutator(c, self.problem), self.problem))
            else:
                rv.append(c)
        return rv


    def Crossover(self, parents: tuple[list[bool], list[bool]]):
        cand = self._crossover(parents[0], parents[1], self.problem)
        return tuple([self._fixer(cand[0], self.problem),self._fixer(cand[1], self.problem)])


    def Replacement(self, population: list[tuple[list[bool],int]], candidates: list[list[bool]]):
        new_population = population
        for c in candidates:
            new_population.append(tuple([c, self._evaluator(c, self.problem)]))

        new_population.sort(key=lambda v: v[1])
        return new_population[:self._populationSize]


    def ParentSelection(self, population: list[tuple[list[bool],int]]):
        self._currentIteration += 1

        (idxa, idxb) = self._parentsSelector(population)
        (a,_) = population[idxa]
        (b,_) = population[idxb]
        return tuple([a,b])


    def TerminationCondition(self):
        if self._currentIteration >= self._generationsNumber:
            return True
        return False


    def InitialPopulation(self):
        new_population =  self._populationInitializer(self._populationSize, self._chromosomeLength, self.problem, self._evaluator)
        new_population.sort(key=lambda v: v[1])

        # for p in new_population:
        #     print("\n")
        #     for f in range(len(p[0])):
        #         if p[0][f]: print(" ", f)
            
        return new_population

    def bestIndividual(self, population: list[tuple[list[bool],int]]):
        if self.best_solution == None or self.best_solution[1] > population[0][1]:
            self.best_solution = population[0]
        print(population[0][1])
        
        fireman = []
        for j in range(self._chromosomeLength):
            if population[0][0][j]:
                fireman.append(j)
        print(fireman)
        return fireman

