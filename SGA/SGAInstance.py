from ProblemDef import FirefighterProblem 
import numpy as np


class SGAInstance:
    def __init__(self, file_name, populationInitializer, populationSize, generationsNumber, parentsSelector, crossover, mutator, fixer, mutationProb, evaluator):
        self.problem: FirefighterProblem = FirefighterProblem.load_from_file(file_name)
        self._populationInitializer = populationInitializer
        self._populationSize = populationSize
        self._chromosomeLength = self.problem.graph.number_of_nodes()
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
                rv.append(self._fixer(self._mutator(c)))
            else:
                rv.append(c)
        return rv
    
    
    def Crossover(self, parents: tuple[list[bool], list[bool]]):
        cand = self._crossover(parents[0], parents[1])
        return tuple[self._fixer(cand[0]),self._fixer(cand[1])]
    
    
    def Replacement(self, population: list[tuple[list[bool],int]], candidates: list[list[bool]]):
        new_population = population
        for c in candidates:
            new_population.append(tuple[c, self._evaluator(c)])
        
        new_population.sort(key=lambda v: v[1])
        return new_population[:self._populationSize]
    
    def ParentSelection(self, population: list[list[bool],int]):
        self._currentIteration += 1
        
        idxs : tuple[int,int] = self._parentsSelector(population)
        return tuple[population[idxs[0]][0],population[idxs[1]][0]]

    
    def TerminationCondition(self):
        if self._currentIteration >= self._generationsNumber:
            return True
        return False

    
    def InitialPopulation(self):
        return self._populationInitializer(self._populationSize, self._chromosomeLength)

    
    def bestIndividual(self, population: list[tuple[list[bool],int]]):
        if self.best_solution == None or self.best_solution[1] > population[0][1]:
            self.best_solution = population[0]
        return population[0][0]

