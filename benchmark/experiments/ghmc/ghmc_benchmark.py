import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.integrators import CustomizableGHMC
from benchmark.plotting import generate_figure_filename

from simtk import unit
import simtk.openmm as mm
import numpy as np
from tqdm import tqdm
import itertools

from openmmtools.testsystems import AlanineDipeptideVacuum, SrcImplicit, SrcExplicit
from simtk.openmm import app
from benchmark.integrators import condense_splitting, generate_sequential_BAOAB_string, generate_all_BAOAB_permutation_strings
from benchmark.utilities import print_array
from benchmark import simulation_parameters

def estimate_acceptance_rate(scheme, timestep, test_system, n_samples=500):
    """Estimate the average acceptance rate for the scheme by drawing `n_samples`
    samples from equilibrium, generating 1-step proposals for each, and averaging
    the acceptance ratios.
    """
    acc_ratios = []

    temperature = test_system.temperature
    ghmc = CustomizableGHMC(splitting=scheme, temperature=temperature, timestep=timestep)
    sim = app.Simulation(test_system.top, test_system.sys, ghmc, test_system.platform)

    for _ in range(n_samples):
        sim.context.setPositions(test_system.draw_sample())
        sim.context.setVelocitiesToTemperature(temperature)

        sim.step(1)

        acc_ratio = ghmc.getGlobalVariableByName("acc_ratio")
        acc_ratios.append(min(1.0, np.nan_to_num(acc_ratio)))

    return np.mean(acc_ratios)

def sweep_over_timesteps(scheme, timesteps, test_system, n_samples=50):
    """If we reach a timestep with a 0.0 accept rate, then don't try
    any subsequent timesteps."""

    acceptance_rates = []
    for timestep in timesteps:

        if len(acceptance_rates) > 0 and acceptance_rates[-1] == 0:
            acceptance_rates.append(0)
        else:
            acceptance_rates.append(estimate_acceptance_rate(scheme, timestep, test_system, n_samples))

    return np.array(acceptance_rates)


def comparison(schemes, timesteps, test_system, n_samples=500):
    curves = dict()
    print(print_array(timesteps))
    for (name, scheme) in schemes:
        curve = sweep_over_timesteps(scheme, timesteps * unit.femtosecond, test_system, n_samples)
        curves[scheme] = curve
        print(name)
        print("\t" + print_array(100 * curve))
        #plt.plot(timesteps, curve, label=name)
    #plt.xlabel("Timestep (fs)")
    #plt.ylabel("GHMC acceptance rate")
    return curves


if __name__ == "__main__":

    # define system and draw equilibrium samples
    temperature = simulation_parameters["temperature"]

    #test_system = get_src_implicit_test_system(temperature)
    test_system = get_alanine_test_system(temperature)

    n_samples_per_timestep = 10000
    timesteps = np.linspace(0.1, 15, 20)

    # comparison of BAOAB-like schemes with permuted force evaluation orders
    schemes = [("Baseline", "V R O R V")] + generate_all_BAOAB_permutation_strings(test_system.sys.getNumForces())


    def plot_curves(schemes, curves):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #areas_under_curves = [np.trapz(c) for c in curves.values()]

        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(schemes))])

        for (name, scheme) in schemes:

            ax.plot(timesteps, curves[scheme], label=name)

        ax.set_xlabel("Timestep (fs)")
        ax.set_ylabel("GHMC acceptance rate")

        lgd = ax.legend(loc=(1, 0), fancybox=True)
        return fig, ax, lgd

    curves = comparison(schemes, timesteps, test_system, n_samples=n_samples_per_timestep)

    fig, ax, lgd = plot_curves(schemes, curves)
    ax.set_title("BAOAB symmetric permutation schemes")
    plt.savefig(generate_figure_filename("baoab_symmetric_perm_comparison.pdf"),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # comparison of BAOAB-like schemes with permuted force evaluation orders
    schemes = [("Baseline", "V R O V R")] + generate_all_BAOAB_permutation_strings(test_system.sys.getNumForces(), symmetric=False)
    curves = comparison(schemes, timesteps, test_system, n_samples=n_samples_per_timestep)

    fig, ax, lgd = plot_curves(schemes, curves)
    ax.set_title("BAOAB asymmetric permutation schemes")
    plt.savefig(generate_figure_filename("baoab_asymmetric_perm_comparison.pdf"),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # it looks like there are 6 clusters -- what are the properties of these clusters?



    # comparison of BAOAB-based MTS schemes
    # mts_schemes = []
    # bond_steps_range = range(1, 6)
    # angle_steps_range = range(1, 11)
    # for i in bond_steps_range:
    #     for j in angle_steps_range:
    #         mts_schemes.append(((i, j), condense_splitting(generate_baoab_mts_string_from_ratios(bond_steps_per_angle_step=i, angle_steps=j))))
    #
    # timesteps = np.linspace(0.1, 100, 20)
    # plt.figure()
    # curves = comparison(mts_schemes, timesteps, test_system, n_samples=n_samples_per_timestep)
    # plt.title("BAOAB MTS schemes\n(bond_steps_per_angle_step, angle_steps_per_outer_timestep)")
    # plt.legend(loc=(1, 0), fancybox=True)
    # plt.savefig("baoab_mts_comparison.pdf")
    # plt.close()
    #
    # # for this comparison, we can also construct a heatmap of the largest allowable timestep
    # # vs. each of the two parameters
    # threshold = 0.90
    #
    # def max_allowable_timestep(curve, threshold):
    #     return timesteps[np.argmax(curve < threshold)]
    #
    #
    # # dictionary mapping from (i,j) tuple to performance curve
    # mts_scheme_dict = dict(zip([scheme[0] for scheme in mts_schemes],
    #                            [curves[scheme[1]] for scheme in mts_schemes]))
    # print(mts_scheme_dict.keys())
    #
    # heat_map = np.zeros((len(bond_steps_range), len(angle_steps_range)))
    # for i in range(len(bond_steps_range)):
    #     for j in range(len(angle_steps_range)):
    #         heat_map[i, j] = max_allowable_timestep(mts_scheme_dict[(bond_steps_range[i], angle_steps_range[j])],
    #                                                 threshold=threshold)
    #
    #
    #
    # plt.figure()
    # plt.imshow(heat_map.T, interpolation="none", cmap="Blues");
    # plt.xlabel("Bond steps per angle step")
    # plt.ylabel("Angle steps per outer step")
    # plt.title("Maximum outer timestep that retains {}% acceptance".format(100*threshold))
    # plt.colorbar()
    # plt.savefig("allowable_timestep_comparison.jpg", dpi=300)
    # plt.close()

    # # comparison of g-BAOAB schemes
    # gbaoab_schemes = []
    # for i in range(1, 4):
    #     gbaoab_schemes.append(("K_r={}".format(i), generate_gbaoab_string(i)))
    # timesteps = np.linspace(0.1, 2.5, 10)
    # plt.figure()
    # curves = comparison(gbaoab_schemes, timesteps, test_system, n_samples=n_samples_per_timestep)
    # plt.title("g-BAOAB schemes")
    # plt.legend(loc="best", fancybox=True)
    # plt.savefig("gbaoab_comparison.pdf")
    #
    # plt.close()