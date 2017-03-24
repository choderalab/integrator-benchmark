import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

from simtk import unit
import simtk.openmm as mm
import numpy as np

def strip_unit(quantity):
    """Take a unit'd quantity and return just its value."""
    return quantity.value_in_unit(quantity.unit)

def stderr(array):
    """Compute the standard error of an array."""
    return np.std(array) / np.sqrt(len(array))

def summarize(array):
    """Given an array, return a string with mean +/- 1.96 * standard error"""
    return "{:.3f} +/- {:.3f}".format(np.mean(array), 1.96 * stderr(array))

def get_summary_string(result, linebreaks=True):
    """Unpack a "result" tuple and return a summary string."""
    W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoint = result
    if linebreaks: separator = "\n\t"
    else: separator = ", "

    summary_string = separator.join(
        ["DeltaF_neq = {:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(sq_uncertainty)),
        "<W_F> = {}".format(summarize(np.array(W_shads_F)[:, -1])),
        "<W_midpoint> = {}".format(summarize(W_midpoint)),
        "<W_R> = {}".format(summarize(np.array(W_shads_R)[:, -1]))])
    return summary_string

def unpack_trajs(result):
    """Unpack a result tuple into x and y coordinates suitable for plotting."""
    W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoint = result
    Fs = []
    Rs = []

    for i in range(len(W_shads_F)):
        traj_F = W_shads_F[i]
        traj_R = W_shads_R[i]# + traj_F[-1]

        x_F = (np.arange(len(traj_F)))
        x_R = (np.arange(len(traj_R))) + x_F[-1]

        Fs.append(traj_F)
        Rs.append(traj_R)

    return x_F, x_R, Fs, Rs

def print_array(array, decimal_places=3):
    format_string = "{:." + str(decimal_places) + "f}"
    return "[" + ", ".join([format_string.format(i) for i in array]) + "]"
