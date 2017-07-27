import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.integrators import CustomizableGHMC
from benchmark.plotting import generate_figure_filename

from simtk import unit
import numpy as np

from benchmark.testsystems import alanine_constrained, alanine_unconstrained, dhfr_constrained
from simtk.openmm import app
from benchmark.utilities import print_array
from tqdm import tqdm

from pickle import dump
import os
from benchmark import DATA_PATH

import seaborn

def estimate_acceptance_rate(scheme, timestep, collision_rate, test_system, n_samples=500):
    """Estimate the average acceptance rate for the scheme by drawing `n_samples`
    samples from equilibrium, generating 1-step proposals for each, and averaging
    the acceptance ratios.
    """
    acc_ratios = []

    temperature = test_system.temperature
    ghmc = CustomizableGHMC(splitting=scheme, temperature=temperature, timestep=timestep, collision_rate=collision_rate)
    sim = app.Simulation(test_system.topology, test_system.system, ghmc, test_system.platform)

    for _ in range(n_samples):
        sim.context.setPositions(test_system.sample_x_from_equilibrium())
        sim.context.setVelocitiesToTemperature(temperature)

        sim.step(1)

        acc_ratio = ghmc.getGlobalVariableByName("acc_ratio")
        acc_ratios.append(min(1.0, np.nan_to_num(acc_ratio)))

    return np.mean(acc_ratios)

def sweep_over_timesteps(scheme, timesteps, collision_rate, test_system, n_samples=50):
    """If we reach a timestep with a 0.0 accept rate, then don't try
    any subsequent timesteps."""

    acceptance_rates = []
    for timestep in tqdm(timesteps):

        if len(acceptance_rates) > 0 and acceptance_rates[-1] == 0:
            acceptance_rates.append(0)
        else:
            acceptance_rates.append(estimate_acceptance_rate(scheme, timestep, collision_rate, test_system, n_samples))

    return np.array(acceptance_rates)


def comparison(schemes, timesteps, collision_rate, test_system, n_samples=500):
    curves = dict()
    print(print_array(timesteps))
    print("Collision_rate: {}".format(collision_rate))
    for (name, scheme) in schemes:
        curve = sweep_over_timesteps(scheme, timesteps * unit.femtosecond, collision_rate, test_system, n_samples)
        curves[scheme] = (timesteps, curve)
        print(name)
        print("\t" + print_array(100 * curve))
    return curves


if __name__ == "__main__":

    #test_system = alanine_constrained
    #test_system_name = "AlanineDipeptideVacuum+constraints"
    test_system = alanine_unconstrained
    test_system_name = "AlanineDipeptideVacuum"
    #test_system = dhfr_constrained
    #test_system_name = "DHFRExplicit+constraints"

    n_samples_per_timestep = 1000
    #timesteps = np.arange(0.1, 7.5, 0.1)
    timesteps = np.arange(0.1, 4.001, 0.1)

    schemes = [("OVRVO", "O V R V O"),
              ("ORVRO", "O R V R O"),
              ("RVOVR", "R V O V R"),
              ("VRORV", "V R O R V"),
              ]

    collision_rates = [("low friction", 1.0 / unit.picosecond),
                       ("high friction", 91.0 / unit.picosecond)]

    def plot_curves(schemes, curves):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(schemes))])

        for (name, scheme) in schemes:

            ax.plot(*curves[scheme], label=name)

        ax.set_xlabel("Timestep (fs)")
        ax.set_ylabel("GHMC acceptance rate")

        lgd = ax.legend(loc=(1, 0), fancybox=True)
        return fig, ax, lgd

    for name, collision_rate in collision_rates:
        curves = comparison(schemes, timesteps, collision_rate, test_system, n_samples=n_samples_per_timestep)

        with open(os.path.join(DATA_PATH, "ghmc_curves_comparison ({}, {}).pkl".format(test_system_name, name)), "wb") as f:
            dump(curves, f)
        fig, ax, lgd = plot_curves(schemes, curves)
        ax.set_title("{} GHMC scheme comparison ({})".format(test_system_name, name))
        #plt.yscale('log')
        plt.savefig(generate_figure_filename("ghmc_scheme_comparison ({}, {}).pdf".format(test_system_name, name)),
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()