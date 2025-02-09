from MTGA.MTGA import *

from MTGA.MTGAInstance import Instance
import MTGA.Generators as Generators
import MTGA.Mutators as Mutators
import MTGA.WeightGenerators as WeightGenerators
import MTGA.GeneMutators as GeneMutators
import Fixers as Fixers
import Evaluators as Evaluators
from ProblemDef import FirefighterProblem

from MTGA.GeneVisualizer import GeneEvolutionRenderer
from Displayer import Displayer

from Utils import *

def correct_solutions(instance):
    evaluator = Evaluators.MainEvaluator()

    solutions = [
        (np.where(bool_array)[0].tolist(), evaluator(instance.problem, bool_array))
        for bool_array, _ in instance.solutions
    ]
    return solutions

def draw_graph(problem: FirefighterProblem, **kwargs):
    node_colors = kwargs.get(
        "node_colors",
        {
            "guarded": "blue",
            "burned": "brown",
            "on_fire": "red",
            "starting": "yellow",
            "default": "green",
        },
    )
    teams = kwargs.get("teams", [])
    colors = []

    # Initialize attributes
    for node in problem.graph.nodes:
        if node in problem.fire_starts:
            colors.append(node_colors["starting"])
        elif node in teams:
            colors.append(node_colors["guarded"])
        else:
            colors.append(node_colors["default"])

    node_size = kwargs.get("node_size", 800)
    font_size = kwargs.get("font_size", 10)
    pos = nx.spring_layout(problem.graph)
    nx.draw(
        problem.graph,
        pos=pos,
        with_labels=True,
        node_color=colors,
        node_size=node_size,
        font_size=font_size,
    )

def collect_and_visualize(problem_file, local_step_size, sigma, **kwargs):
    instance = Instance(problem_file,
        evaluator=Evaluators.CummulativeEvaluator(),
        mutator=Mutators.WalkMutator(),
        generator=Generators.TruncatedNormal(),
        fixer=Fixers.ChoiceFixer(),
        weight_generator=WeightGenerators.SoftmaxWeights(),
        gene_mutator=GeneMutators.Normal(sigma=sigma))

    tribe_number = kwargs.get('tribe_number', 8)
    tribe_population = kwargs.get('tribe_population', 30)
    number_of_iterations = kwargs.get('number_of_iterations', 200)
    print_updates = kwargs.get('print_updates', False)
    record_metrics = kwargs.get('record_metrics', False)
    
    data = optimize_and_collect(instance,
                                tribe_number=tribe_number,
                                tribe_population=tribe_population,
                                number_of_iterations=number_of_iterations,
                                print_updates=print_updates,
                                record_metrics=record_metrics,
                                local_step_size=local_step_size,
                                )
    print("data collected")
    visualize_gene_evolution(data)
    return data, instance
    
