from MTGA.MTGA import *

from MTGA.MTGAInstance import Instance
import MTGA.Generators as Generators
import MTGA.Mutators as Mutators
import MTGA.WeightGenerators as WeightGenerators
import MTGA.GeneMutators as GeneMutators
import Fixers as Fixers
import Evaluators as Evaluators

from MTGA.GeneVisualizer import GeneEvolutionRenderer
from Displayer import Displayer

from Utils import *


instance = Instance(
    "problems/p1.json",
    evaluator=Evaluators.CummulativeEvaluator(),
    mutator=Mutators.WalkMutator(),
    generator=Generators.TruncatedNormal(),
    fixer=Fixers.ChoiceFixer(),
    weight_generator=WeightGenerators.SoftmaxWeights(),
    gene_mutator=GeneMutators.Normal(),
)

data = optimize_and_collect(
    instance, number_of_iterations=7, tribe_number=3, tribe_population=20
)
# data = optimize(instance)

# vis = GeneEvolutionRenderer(independent_ylim=False)
# vis.visualize(data,
#     output_path="output/p2_evolution.gif")
