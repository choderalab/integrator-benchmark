from benchmark.integrators import LangevinSplittingIntegrator
import numpy as np
from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass_amber

from benchmark.testsystems import t4_constrained
test_system = t4_constrained

from copy import deepcopy
topology = test_system.topology
default_system = deepcopy(test_system.system)

from simtk import unit

def integrator_factory(dt):
    return LangevinSplittingIntegrator(collision_rate=91.0 / unit.picosecond, timestep=dt)


def test_stability(test_system, simulation, n_samples, trajectory_length):
    for _ in range(n_samples):
        x = test_system.sample_x_from_equilibrium()
        v = test_system.sample_v_given_x(x)

        simulation.context.setPositions(x)
        simulation.context.setVelocities(v)

        simulation.step(trajectory_length)

        if np.isnan(simulation.context.getState(getPositions=True).getPositions(asNumpy=True)).sum() > 0:
            return False

    return True


def find_stability_threshold_deterministic_search(test_system, integrator_factory, max_dt=10 * unit.femtoseconds, max_bisections=15,
                             n_samples=10, trajectory_length=10000):
    min_dt = 0 * unit.femtoseconds

    print("Searching for maximum stable timestep in range: [{}, {}]".format(min_dt, max_dt))
    for _ in range(max_bisections):
        dt = (max_dt + min_dt) / 2
        simulation = test_system.construct_simulation(integrator_factory(dt))
        stable = test_stability(test_system, simulation, n_samples, trajectory_length)

        if stable:
            min_dt = dt
            print("\t{} stable! Current range: [{}, {}]".format(dt, min_dt, max_dt))
        else:
            max_dt = dt
            print("\t{} unstable! Current range: [{}, {}]".format(dt, min_dt, max_dt))

        if min_dt == max_dt:
            break

    print("Final range for stability threshold: [{}, {}]\n\n".format(min_dt, max_dt))
    return min_dt, max_dt


from tqdm import tqdm


def probabilistic_bisection(noisy_oracle, initial_limits=(0, 1), p=0.6, n_iterations=1000, resolution=100000):
    """

    Implementation details:
    * For convenience / ease of implementation, represents the belief pdf numerically
    """

    x = np.linspace(initial_limits[0], initial_limits[1], resolution)
    f = np.ones(len(x))
    f /= np.trapz(f, x)

    fs = [f]

    zs = []
    for _ in tqdm(range(n_iterations)):
        f = fs[-1]

        median = x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - 0.5))]
        z = noisy_oracle(median)
        zs.append(z)

        new_f = np.array(f)

        if z > 0:
            new_f[np.where(x >= median)] *= p
            new_f[np.where(x < median)] *= (1 - p)
        else:
            new_f[np.where(x >= median)] *= (1 - p)
            new_f[np.where(x < median)] *= p

        new_f /= np.trapz(new_f, x)

        fs.append(new_f)
    return x, zs, fs

def get_hmr_stability_threshold_curve(splitting="V R O R V", traj_length=1000):
    h_masses = np.arange(0.5, 4.51, 0.25)
    stability_thresholds = []

    for h_mass in h_masses:
        hmr_system = repartition_hydrogen_mass_amber(topology, default_system,
                                                 scale_factor=h_mass)
        test_system.system = hmr_system
        lsi = LangevinSplittingIntegrator(splitting)
        simulation = test_system.construct_simulation(lsi)

        def stability_oracle(dt):
            simulation.integrator.setStepSize(dt * unit.femtoseconds)
            stable = test_stability(test_system, simulation, 1, traj_length)
            if stable:
                return 1
            else:
                return -1

        x, _, fs_heavy_H = probabilistic_bisection(stability_oracle, initial_limits=(0, 10), p=0.6, n_iterations=100)
        stability_thresholds.append(x[np.argmax(fs_heavy_H[-1])])
    return h_masses, stability_thresholds


if __name__ == "__main__":
    ys = {}
    for scheme in ["V R O R V", "R V O V R", "O V R V O", "O R V R O"]:
        print(scheme)
        h_masses, stability_thresholds = get_hmr_stability_threshold_curve(scheme)
        ys[scheme] = stability_thresholds
        print("\tStability thresholds: {}".format(stability_thresholds))
        print("\tBest H-mass tested: {}".format(h_masses[np.argmax(ys[scheme])]))

    from pickle import dump
    #import os
    #from benchmark import DATA_PATH
    #with open(os.path.join(DATA_PATH, "stability_limits_hmr.pkl"), "wb") as f:
    #    dump((h_masses, ys), f)
    with open("stability_limits_hmr.pkl", "wb") as f:
        dump((h_masses, ys), f)
