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

    if kwargs.get('print updates', True):
        print('[%s][%3d] %14.8f %14.8f {%12.8f %12.8f %12.8f %12.8f}' %
              (ftime, iteration, run_time, iter_time, iter_min, iter_mean, iter_max, iter_std))

    if kwargs.get('record metrics', True):
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

def check_shape(arr, shape, msg=None):
    if arr.shape != shape:
        if msg == None:
            print(f"expected shape {shape}, but gotten {arr.shape}")
        else:
            print(f"{msg} expected shape {shape}, but gotten {arr.shape}")
