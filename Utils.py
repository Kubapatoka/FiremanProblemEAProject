import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

from datetime import datetime

def pp_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def register_metrics(metrics, iteration, run_time, iter_time, objective_value, **kwargs):
    ftime = pp_now()
    iter_min  = objective_value.min()
    iter_mean = objective_value.mean()
    iter_max  = objective_value.max()
    iter_std  = objective_value.std()

    if kwargs.get('print_updates', True):
        print('[%s][%3d] %14.8f %14.8f {%12.8f %12.8f %12.8f %12.8f}' %
              (ftime, iteration, run_time, iter_time, iter_min, iter_mean, iter_max, iter_std))

    if kwargs.get('record_metrics', True):
        new_row = {
            "iteration": iteration,
            "time": iter_time,
            "min":  iter_min,
            "mean": iter_mean,
            "max":  iter_max,
            "std":  iter_std
        }
        metrics.loc[len(metrics)] = new_row


def draw_progress(register, title):
    npregister = np.array(register)
    print(npregister.shape)

    plt.figure()
    plt.title(title)
    labels = ['min', 'mean', 'max', 'std', 'time', ]
    colors = list(pltcolors.TABLEAU_COLORS.values())
    for i, c in enumerate(colors[:3]):
        plt.plot(npregister[:,i], color=c, label=labels[i])
    plt.legend()
    plt.show()
    
    i = 3
    plt.figure()
    plt.title(title+" - std")
    plt.plot(npregister[:,i], color=c, label=labels[i])
    plt.legend()
    plt.show()
    
    i = 4
    plt.figure()
    plt.title(title+" - time")
    plt.plot(npregister[:,i], label=labels[i])
    plt.legend()
    plt.show()


def softmax(array, temperature=5):
    max_value = np.max(array)
    scaled_values = (array - max_value) / temperature
    exp_values = np.exp(scaled_values)
    weights = exp_values / np.sum(exp_values)
    return weights

def sigmoid(array, steepness):
    final_steepness = steepness #np.std(array, axis=-1) * steepness
    final_midpoint  = np.mean(array, axis=-1)
    sigmoid_values = 1/(1 + np.exp(-final_steepness * (array - final_midpoint)))
    return sigmoid_values / np.sum(sigmoid_values, axis=-1)

def smooth_clip(x, lower_bound=0, steepness=200):
    return (1/steepness) * np.log(1 + np.exp(steepness * (x-lower_bound))) + lower_bound

def check_shape(arr, shape, msg=None):
    if arr.shape != shape:
        if msg == None:
            print(f"expected shape {shape}, but gotten {arr.shape}")
        else:
            print(f"{msg} expected shape {shape}, but gotten {arr.shape}")
            
            
def genotypeToFenotype(gen : list[bool]):
    fireman = []
    for j in range(len(gen)):
        if gen[j]:
            fireman.append(j)
    return fireman

def fenotypeToGenotype(fen : list[int], chromosomeLen: int):
    fireman = [False for _ in range(chromosomeLen)]
    for j in fen:
        fireman[j] = True
    return fireman

