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

print('OpenMM version: ', mm.version.full_version)


class TestSystem():
    def __init__(self, samples, temperature, mm_topology, mm_sys, mm_platform):
        self.samples = samples
        self.temperature = temperature
        self.top = mm_topology
        self.sys = mm_sys
        self.platform = mm_platform

    def draw_sample(self):
        return self.samples[np.random.randint(len(self.samples))]


from openmmtools.integrators import GHMCIntegrator, GradientDescentMinimizationIntegrator


def get_equilibrium_samples(topology, system, positions,
                            platform, temperature,
                            burn_in_length, n_samples, thinning_interval,
                            ghmc_timestep=1.5 * unit.femtoseconds, **kwargs):
    """Minimize energy for 100 steps, run GHMC for burn_in_length steps, then
    run GHMC for thinning_interval * n_samples steps, storing snapshots every
    thinning_interval frames.

    Return list of configuration samples and the GHMC simulation.
    """

    # Minimize energy by gradient descent
    print("Minimizing...")
    minimizer = GradientDescentMinimizationIntegrator()
    min_simulation = app.Simulation(topology, system, minimizer, platform)
    min_simulation.context.setPositions(positions)
    min_simulation.context.setVelocitiesToTemperature(temperature)
    min_simulation.step(100)

    # Define unbiased simulation...
    ghmc = GHMCIntegrator(temperature, timestep=ghmc_timestep)
    get_acceptance_rate = lambda: ghmc.getGlobalVariableByName("naccept") / ghmc.getGlobalVariableByName("ntrials")
    unbiased_simulation = app.Simulation(topology, system, ghmc, platform)
    unbiased_simulation.context.setPositions(positions)
    unbiased_simulation.context.setVelocitiesToTemperature(temperature)

    # Equilibrate
    print('"Burning in" unbiased GHMC sampler for {:.3}ps...'.format(
        (burn_in_length * ghmc_timestep).value_in_unit(unit.picoseconds)))
    unbiased_simulation.step(burn_in_length)
    print("Burn-in GHMC acceptance rate: {:.3f}%".format(100 * get_acceptance_rate()))
    ghmc.setGlobalVariableByName("naccept", 0)
    ghmc.setGlobalVariableByName("ntrials", 0)

    # Collect equilibrium samples
    print("Collecting equilibrium samples...")
    equilibrium_samples = []
    for _ in tqdm(range(n_samples)):
        unbiased_simulation.step(thinning_interval)
        equilibrium_samples.append(
            unbiased_simulation.context.getState(getPositions=True).getPositions(asNumpy=True))
    print("Equilibrated GHMC acceptance rate: {:.3f}%".format(100 * get_acceptance_rate()))

    return equilibrium_samples, unbiased_simulation


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


def print_array(array, decimal_places=3):
    format_string = "{:." + str(decimal_places) + "f}"
    return "[" + ", ".join([format_string.format(i) for i in array]) + "]"


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


def keep_only_some_forces(system):
    """Remove unwanted forces, e.g. center-of-mass motion removal"""
    forces_to_keep = ["HarmonicBondForce", "HarmonicAngleForce",
                      "PeriodicTorsionForce", "NonbondedForce"]
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ not in forces_to_keep:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        print('   Removing %s' % system.getForce(force_index).__class__.__name__)
        system.removeForce(force_index)


def load_alanine(constrained=True):
    """Load AlanineDipeptide vacuum, optionally with hydrogen bonds constrained"""
    if constrained:
        constraints = app.HBonds
    else:
        constraints = None
    testsystem = AlanineDipeptideVacuum(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system)

    return topology, system, positions

def condense_splitting(splitting_string):
    """Since some operators commute, we can simplify some splittings.

    Here, we replace repeated O steps or V{i} steps.

    Splitting is a list of steps.

    Examples:

        O V R V V V O should condense to
        O V R V O

        and O V O O V R V
        should condense to:
        O V R V

        since
    """

    # first split into chunks of either velocity or position updates
    # don't collapse position updates, do collapse velocity updates

    splitting = splitting_string.upper().split()

    equivalence_classes = {"R":{"R"}, "V":{"O", "V"}, "O":{"O", "V"}}

    current_chunk = [splitting[0]]
    collapsed = []

    def collapse_chunk(current_chunk):

        if current_chunk[0] == "R":
            return current_chunk
        else:
            return list(set(current_chunk))

    for i in range(1, len(splitting)):

        # if the next step in the splitting is
        if splitting[i][0] in equivalence_classes[splitting[i-1][0]]:
            current_chunk.append(splitting[i])
        else:
            collapsed += collapse_chunk(current_chunk)
            current_chunk = [splitting[i]]


    collapsed = collapsed + collapse_chunk(current_chunk)

    collapsed_string = " ".join(collapsed)
    if len(collapsed) < len(splitting):
        print("Shortened the splitting from {} steps to {} steps ({} --> {})".format(
            len(splitting), len(collapsed), splitting_string, collapsed_string
        ))

    return collapsed_string


def get_alanine_test_system(temperature):
    top, sys, pos = load_alanine(constrained=True)
    platform = mm.Platform.getPlatformByName("Reference")
    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=1.5 * unit.femtoseconds,
                                                           burn_in_length=1000, n_samples=1000, thinning_interval=100)
    test_system = TestSystem(samples, temperature, top, sys, platform)
    return test_system

def get_src_implicit_test_system(temperature):
    testsystem = SrcImplicit()
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    platform = mm.Platform.getPlatformByName("OpenCL")
    platform.setPropertyDefaultValue('OpenCLPrecision', 'double')

    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=0.5 * unit.femtoseconds,
                                                           burn_in_length=500, n_samples=500, thinning_interval=5)
    test_system = TestSystem(samples, temperature, top, sys, platform)
    return test_system


def get_src_explicit_test_system(temperature):
    testsystem = SrcExplicit()
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    platform = mm.Platform.getPlatformByName("OpenCL")
    platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')

    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=0.5 * unit.femtoseconds,
                                                           burn_in_length=100, n_samples=100, thinning_interval=5)
    test_system = TestSystem(samples, temperature, top, sys, platform)
    return test_system


def generate_sequential_BAOAB_string(force_group_list, symmetric=True):
    """Generate BAOAB-like schemes that break up the "V R" step
    into multiple sequential updates

    E.g. force_group_list=(0,1,2), symmetric=True -->
        "V0 R V1 R V2 R O R V2 R V1 R V0"
    force_group_list=(0,1,2), symmetric=False -->
        "V0 R V1 R V2 R O V0 R V1 R V2 R"
    """

    VR = []
    for i in force_group_list:
        VR.append("V{}".format(i))
        VR.append("R")


    if symmetric:
        return " ".join(VR + ["O"] + VR[::-1])
    else:
        return " ".join(VR + ["O"] + VR)


def generate_all_BAOAB_permutation_strings(n_force_groups, symmetric=True):
    """Generate all of the permutations of range(n_force_groups), and evaluate their
    acceptance rates
    """
    return [(perm, generate_sequential_BAOAB_string(perm, symmetric)) for perm in itertools.permutations(range(n_force_groups))]

if __name__ == "__main__":

    # define system and draw equilibrium samples
    temperature = 298 * unit.kelvin

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