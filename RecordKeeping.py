import pandas as pd
import numpy as np

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

    if metrics is not None:
        new_row = {
            "iteration": iteration,
            "time": iter_time,
            "min":  iter_min,
            "mean": iter_mean,
            "max":  iter_max,
            "std":  iter_std
        }
        metrics.loc[len(metrics)] = new_row
