import ProblemDef


## WIP

def SGA(instance):
    population = instance.InitialPopulation
    #population evaluation
    
    while not instance.TerminationCondition(population):
        parents = instance.ParentSelection(population)
        candidates = instance.Crossover(parents)
        candidates = instance.Mutation(candidates)
        population = instance.Replacement(population, candidates)
        #population evaluation