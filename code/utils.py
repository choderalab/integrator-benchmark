import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from simtk import unit
import simtk.openmm as mm
import numpy as np
from openmmtools.testsystems import WaterBox
W_unit = unit.kilojoule_per_mole

def configure_platform(platform_name='Reference'):
    """Set precision, etc..."""
    if platform_name.upper() == 'Reference'.upper():
        platform = mm.Platform.getPlatformByName('Reference')
    elif platform_name.upper() == 'OpenCL'.upper():
        platform = mm.Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
    elif platform_name.upper() == 'CUDA'.upper():
        platform = mm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('CUDAPrecision', 'mixed')
    else:
        raise(ValueError("Invalid platform name"))
    return platform

def strip_unit(quantity):
    """Take a unit'd quantity and return just its value."""
    return quantity.value_in_unit(quantity.unit)

def load_waterbox():
    """Load WaterBox test system with non-default PME cutoff and error tolerance... """
    testsystem = WaterBox(constrained=True, ewaldErrorTolerance=1e-5, cutoff=10*unit.angstroms)
    (system, positions) = testsystem.system, testsystem.positions
    #positions = np.load("waterbox.npy") # load equilibrated configuration saved beforehand
    return system, positions

def get_total_energy(simulation):
    """Compute the kinetic energy + potential energy of the simulation."""
    state = simulation.context.getState(getEnergy=True)
    ke, pe = state.getKineticEnergy(), state.getPotentialEnergy()
    return ke + pe

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
        traj_R = W_shads_R[i] + traj_F[-1] + W_midpoint[i]

        x_F = (np.arange(len(traj_F)))
        x_R = (np.arange(len(traj_R))) + x_F[-1]

        Fs.append(traj_F)
        Rs.append(traj_R)

    return x_F, x_R, Fs, Rs

def plot(results, name=""):
    """Given a results dictionary that maps scheme strings to work trajectories / DeltaFNeq estimates, plot
    the trajectories and their averages, and print a summary."""
    schemes = results.keys()

    # get min and max
    y_min, y_max = np.inf, -np.inf
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])
        y_min_ = min(np.min(Fs), np.min(Rs))
        y_max_ = max(np.max(Fs), np.max(Rs))
        if y_min_ < y_min: y_min = y_min_
        if y_max_ > y_max: y_max = y_max_

    # plot individuals
    for scheme in schemes:

        # plot the raw shadow work trajectories
        plt.figure()
        plt.title(scheme + get_summary_string(results[scheme], linebreaks=False))

        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        traj_style = {"linewidth": 0.1, "alpha": 0.3}
        for i in range(len(Fs)):
            plt.plot(x_F, Fs[i], color='blue', **traj_style)
            plt.plot(x_R, Rs[i], color='green', **traj_style)

        plt.ylim(y_min, y_max)
        plt.xlabel('# steps')
        plt.ylabel('Shadow work')
        plt.savefig('{}_work_trajectories_{}.jpg'.format(name, scheme), dpi=300)

        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color="blue")

        plt.plot(x_R, R_mean, color="green")

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color="blue", alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color="green", alpha=0.3)

        plt.savefig('{}_averaged_work_trajectories_{}.jpg'.format(name, scheme), dpi=300)
        plt.close()
    # also make a comparison figure with all of the integrators, just with the confidence bands
    # instead of the full trajectories
    colors = dict(zip(schemes, "blue green orange purple darkviolet".split()))
    plt.figure()
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        # plot the averages
        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color=colors[scheme], label=scheme)
        plt.plot(x_R, R_mean, color=colors[scheme])

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color=colors[scheme], alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color=colors[scheme], alpha=0.3)

        # plot vertical line where midpoint operator is applied
        # plt.vlines((x_F[-1] + x_R[0])/2.0, y_min, y_max, linestyles='--', color='grey')

    plt.xlabel('# steps')
    plt.ylabel('Shadow work')
    plt.title('Comparison')
    plt.legend(fancybox=True, loc='best')
    plt.savefig('{}_averaged_work_trajectories_comparison.jpg'.format(name), dpi=300)
    plt.close()