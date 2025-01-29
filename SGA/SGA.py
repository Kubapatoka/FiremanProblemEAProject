import ProblemDef


# WIP

def SGA(instance):
    population = instance.InitialPopulation()
    # population is a sorted list of pairs: (genotype, fitness_value)
    
    while not instance.TerminationCondition():
        parents = instance.ParentSelection(population)
        # candidates are genotype
        candidates = instance.Crossover(parents)
        candidates = instance.Mutation(candidates)
        population = instance.Replacement(population, candidates)
    return instance.bestIndividual(population)