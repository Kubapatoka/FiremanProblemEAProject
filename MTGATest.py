from MTGA.MTGA import *

from MTGA.MTGAInstance import Instance
import MTGA.Generators as Generators
import MTGA.Mutators as Mutators
import MTGA.WeightGenerators as WeightGenerators
import Fixers as Fixers
import Evaluators as Evaluators

from MTGA.GeneVisualizer import GeneEvolutionRenderer

from ProblemDef import FirefighterProblem
from Utils import *


instance = Instance("problems/p1.json",
    evaluator=Evaluators.CummulativeEvaluator(),
    mutator=Mutators.WalkMutator(),
    generator=Generators.Generator(),
    fixer=Fixers.ChoiceFixer(),
    weight_generator=WeightGenerators.SoftmaxWeights())

data = optimize_and_collect(instance)
# data = optimize(instance)

vis = GeneEvolutionRenderer("output/p1_evolution.gif")
vis.visualize(data)
