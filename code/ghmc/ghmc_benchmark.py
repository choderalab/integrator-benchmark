import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integrators import CustomizableGHMC
from simtk import unit
import simtk.openmm as mm
import numpy as np
from tqdm import tqdm

from openmmtools.testsystems import AlanineDipeptideVacuum
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
        plt.plot(timesteps, curve, label=name)
    plt.xlabel("Timestep (fs)")
    plt.ylabel("GHMC acceptance rate")
    return curves


def generate_gbaoab_string(K_r=1):
    """K_r=1 --> 'V R O R V
    K_r=2 --> 'V R R O R R V'
    etc.
    """
    Rs = ["R"] * K_r
    return " ".join(["V"] + Rs + ["O"] + Rs + ["V"])


def generate_baoab_mts_string(groups, K_r=1):
    """Multi timestep generalization of the solvent-solute splitting scheme presented above...

    In the solvent-solute splitting, we have a "fast" group and a "slow" group.
    What if we have more than two groups?

    In the the straightforard generalization of the solvent-solute scheme, we do something like this:

    Accept groups, a list of 2-tuples, where each tuple contains an iterable of force group indices and
    an execution-frequency ratio.

    For example, groups=[([0], 1), ([1], 2), ([2], 2)] should be taken to mean:
        execute V1 twice as often as V0, and execute V2 twices as often as V1....

        To be concrete:

        If groups=[([0], 1), ([1], 2), ([2], 2)], K_r=1 we would have:

                V0 (V1 (V2 R^K_r O R^K_r V2)^2 V1)^2 V0

    """
    Rs = ["R"] * K_r

    ratios = [group[1] for group in groups]
    forces = [["V{}".format(i) for i in group[0]] for group in groups]

    inner_loop_string = (forces[-1] + Rs + ["O"] + Rs + forces[-1]) * ratios[-1]

    for i in range(len(ratios))[::-1][1:]:
        inner_loop_string = (forces[i] + inner_loop_string + forces[i]) * ratios[i]

    return " ".join(inner_loop_string)


def generate_baoab_mts_string_from_ratios(bond_steps_per_angle_step=5, angle_steps=7):
    """Assuming there are just four groups.
    0: Bonds (Cheap)
    1,2: Angles and torsions (let's say that's ~5x more expensive than bonds)
    3: Nonbonded (~wayyyy more expensive than bonded interactions)
    """
    return generate_baoab_mts_string([([3], 1), ([1, 2], angle_steps), ([0], bond_steps_per_angle_step)])


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




if __name__ == "__main__":

    # define system and draw equilibrium samples
    temperature = 298 * unit.kelvin

    top, sys, pos = load_alanine(constrained=True)
    platform = mm.Platform.getPlatformByName("Reference")
    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=1.5 * unit.femtoseconds,
                                                           burn_in_length=1000, n_samples=1000, thinning_interval=100)
    test_system = TestSystem(samples, temperature, top, sys, platform)


    n_samples_per_timestep = 500

    # comparison of BAOAB-based MTS schemes
    mts_schemes = []
    bond_steps_range = range(1, 6)
    angle_steps_range = range(1, 11)
    for i in bond_steps_range:
        for j in angle_steps_range:
            mts_schemes.append(((i, j), generate_baoab_mts_string_from_ratios(bond_steps_per_angle_step=i, angle_steps=j)))

    timesteps = np.linspace(0.1, 100, 50)
    plt.figure()
    curves = comparison(mts_schemes, timesteps, test_system, n_samples=n_samples_per_timestep)
    plt.title("BAOAB MTS schemes\n(bond_steps_per_angle_step, angle_steps_per_outer_timestep)")
    plt.legend(loc=(1, 0), fancybox=True)
    plt.savefig("baoab_mts_comparison.pdf")
    plt.close()

    # for this comparison, we can also construct a heatmap of the largest allowable timestep
    # vs. each of the two parameters
    threshold = 0.90

    def max_allowable_timestep(curve, threshold):
        return timesteps[np.argmax(curve < threshold)]


    # dictionary mapping from (i,j) tuple to performance curve
    mts_scheme_dict = dict(zip([scheme[0] for scheme in mts_schemes],
                               [curves[scheme[1]] for scheme in mts_schemes]))
    print(mts_scheme_dict.keys())

    heat_map = np.zeros((len(bond_steps_range), len(angle_steps_range)))
    for i in range(len(bond_steps_range)):
        for j in range(len(angle_steps_range)):
            heat_map[i, j] = max_allowable_timestep(mts_scheme_dict[(bond_steps_range[i], angle_steps_range[j])],
                                                    threshold=threshold)



    plt.figure()
    plt.imshow(heat_map.T, interpolation="none", cmap="Blues");
    plt.xlabel("Bond steps per angle step")
    plt.ylabel("Angle steps per outer step")
    plt.title("Maximum outer timestep that retains {}% acceptance".format(100*threshold))
    plt.colorbar()
    plt.savefig("allowable_timestep_comparison.jpg", dpi=300)
    plt.close()

    # comparison of g-BAOAB schemes
    gbaoab_schemes = []
    for i in range(1, 11):
        gbaoab_schemes.append(("K_r={}".format(i), generate_gbaoab_string(i)))
    timesteps = np.linspace(0.1, 10.0, 20)
    plt.figure()
    curves = comparison(gbaoab_schemes, timesteps, test_system, n_samples=n_samples_per_timestep)
    plt.title("g-BAOAB schemes")
    plt.legend(loc="best", fancybox=True)
    plt.savefig("gbaoab_comparison.pdf")

    plt.close()