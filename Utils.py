import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

import copy
from datetime import datetime


def pp_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def register_metrics(
    metrics, iteration, run_time, iter_time, objective_value, **kwargs
):
    ftime = pp_now()
    iter_min = objective_value.min()
    iter_mean = objective_value.mean()
    iter_max = objective_value.max()
    iter_std = objective_value.std()

    if kwargs.get("print_updates", True):
        print(
            "[%s][%3d] %14.8f %14.8f {%12.8f %12.8f %12.8f %12.8f}"
            % (
                ftime,
                iteration,
                run_time,
                iter_time,
                iter_min,
                iter_mean,
                iter_max,
                iter_std,
            )
        )

    if kwargs.get("record_metrics", True):
        new_row = {
            "iteration": iteration,
            "time": iter_time,
            "min": iter_min,
            "mean": iter_mean,
            "max": iter_max,
            "std": iter_std,
        }
        metrics.loc[len(metrics)] = new_row


def visualize_gene_evolution(collected_data):
    first_genes = collected_data[0]["genes"]
    tribe_num, chromosome_len = first_genes.shape

    generations = [entry["generation"] for entry in collected_data]
    gene_data = np.array(
        [entry["genes"] for entry in collected_data]
    )  # Shape: (iterations, tribe_num, chromosome_len)

    for tribe in range(tribe_num):
        plt.figure(figsize=(10, 6))
        for chromosome in range(chromosome_len):
            plt.plot(
                generations,
                gene_data[:, tribe, chromosome],
                label=f"Chromosome {chromosome}",
            )

        plt.title(f"Gene Evolution for Tribe {tribe}")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()


def draw_progress(register, title):
    npregister = np.array(register)
    print(npregister.shape)

    plt.figure()
    plt.title(title)
    labels = [
        "min",
        "mean",
        "max",
        "std",
        "time",
    ]
    colors = list(pltcolors.TABLEAU_COLORS.values())
    for i, c in enumerate(colors[:3]):
        plt.plot(npregister[:, i], color=c, label=labels[i])
    plt.legend()
    plt.show()

    i = 3
    plt.figure()
    plt.title(title + " - std")
    plt.plot(npregister[:, i], color=c, label=labels[i])
    plt.legend()
    plt.show()

    i = 4
    plt.figure()
    plt.title(title + " - time")
    plt.plot(npregister[:, i], label=labels[i])
    plt.legend()
    plt.show()


def softmax(array, temperature=5):
    max_value = np.max(array)
    scaled_values = (array - max_value) / temperature
    exp_values = np.exp(scaled_values)
    weights = exp_values / np.sum(exp_values)
    return weights


def sigmoid(array, steepness):
    final_steepness = steepness  # np.std(array, axis=-1) * steepness
    final_midpoint = np.mean(array, axis=-1)
    sigmoid_values = 1 / (1 + np.exp(-final_steepness * (array - final_midpoint)))
    return sigmoid_values / np.sum(sigmoid_values, axis=-1)


def smooth_clip(x, lower_bound=0, steepness=200):
    return (1 / steepness) * np.log(
        1 + np.exp(steepness * (x - lower_bound))
    ) + lower_bound


def check_shape(arr, shape, msg=None):
    if arr.shape != shape:
        if msg == None:
            print(f"expected shape {shape}, but gotten {arr.shape}")
        else:
            print(f"{msg} expected shape {shape}, but gotten {arr.shape}")


def genotypeToFenotype(gen: list[bool]):
    fireman = []
    for j in range(len(gen)):
        if gen[j]:
            fireman.append(j)
    return fireman


def fenotypeToGenotype(fen: list[int], chromosomeLen: int):
    fireman = [False for _ in range(chromosomeLen)]
    for j in fen:
        fireman[j] = True
    return fireman


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def compute_jsd_matrix(points):
    points = points / points.sum(axis=1, keepdims=True)

    jsd_matrix = squareform(pdist(points, metric=jensen_shannon_divergence))

    return jsd_matrix


def compute_scores(points):
    jsd_matrix = compute_jsd_matrix(points)
    sigma = np.mean(jsd_matrix)
    affinity_matrix = np.exp(-jsd_matrix / sigma)
    scores = np.sum(affinity_matrix, axis=1) - np.diag(affinity_matrix)

    return scores


def print_parameters(file, values, *args):
    for i in range(values.shape[0]):
        file.write(f"Tribe {i}\n")
        for j in range(values.shape[1]):
            file.write(f"- {values[i,j]:3.3f}")
            for arg in args:
                file.write(f" {arg[i,j]:3.3f}")
            file.write("\n")


def print_sorted_parameters(file, values, *args):
    sorted_indices = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, sorted_indices, axis=1)
    sorted_args = []
    for arg in args:
        sorted_args.append(np.take_along_axis(arg, sorted_indices, axis=1))

    print_parameters(file, sorted_values, *sorted_args)


def correct_solutions(instance, evaluator=None):
    def eval(arr):
        if evaluator is None:
            return arr
        else:
            return evaluator(instance.problem, arr)

    solutions = [
        (np.where(bool_array)[0].tolist(), eval(bool_array))
        for bool_array, _ in instance.solutions
    ]
    return solutions


def numpy_reset_default_prints():
    np.set_printoptions(
        edgeitems=3,
        infstr="inf",
        linewidth=75,
        nanstr="nan",
        precision=8,
        suppress=False,
        threshold=1000,
        formatter=None,
    )


def draw_graph(graph, fire_starts, **kwargs):
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
    for node in graph.nodes:
        if node in fire_starts:
            colors.append(node_colors["starting"])
        elif node in teams:
            colors.append(node_colors["guarded"])
        else:
            colors.append(node_colors["default"])

    print("colors", colors)
    print("nodes", graph.nodes)

    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_color=colors,
        node_size=800,
        font_size=10,
    )
