import numpy as np

from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import alanine_constrained
from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass_amber

from copy import deepcopy

from simtk import unit

from benchmark.testsystems import NonequilibriumSimulator
from benchmark.evaluation import estimate_nonequilibrium_free_energy


def integrator_factory(dt):
    return LangevinSplittingIntegrator(collision_rate=91.0 / unit.picosecond, timestep=dt)


def test_stability(test_system, simulation, n_samples, trajectory_length):
    for _ in range(n_samples):
        x = test_system.sample_x_from_equilibrium()
        v = test_system.sample_v_given_x(x)

        simulation.context.setPositions(x)
        simulation.context.setVelocities(v)

        for _ in range(round(trajectory_length / 10)):
            simulation.step(10)

            # check whether energy is nan or positive, rather than whether positions are NaNs
            U = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            if not (U.value_in_unit(U.unit) <= 0):
                return False

    return True


def find_stability_threshold_deterministic_search(test_system, integrator_factory, max_dt=10 * unit.femtoseconds,
                                                  max_bisections=15,
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


def get_hmr_stability_threshold_curve(test_system, splitting="V R O R V", traj_length=1000):
    topology = test_system.topology
    default_system = deepcopy(test_system.system)

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
    test_system = repartition_hydrogen_mass_amber(topology, default_system, scale_factor=1)
    return h_masses, stability_thresholds


def error_oracle_factory(test_system, error_threshold, scheme="O V R V O", n_protocol_samples=100, protocol_length=1000):
    integrator = LangevinSplittingIntegrator(scheme, timestep=1.0 * unit.femtosecond)
    sim = NonequilibriumSimulator(test_system, integrator)

    def error_oracle(dt):
        integrator.setStepSize(dt * unit.femtosecond)

        W_F, W_R = sim.collect_protocol_samples(n_protocol_samples, protocol_length)
        DeltaF_neq, sq_unc = estimate_nonequilibrium_free_energy(W_F, W_R)

        print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(sq_unc)))

        if error_threshold > np.abs(DeltaF_neq):
            return 1
        else:
            return -1

    return error_oracle


if __name__ == "__main__":
    test_system = alanine_constrained
    reference_integrator = LangevinSplittingIntegrator("O V R V O", timestep=2.0 * unit.femtosecond)
    reference_sim = NonequilibriumSimulator(test_system, reference_integrator)

    W_F, W_R = reference_sim.collect_protocol_samples(1000, 1000)
    DeltaF_neq, sq_unc = estimate_nonequilibrium_free_energy(W_F, W_R)

    results = {}
    for scheme in ["V R O R V", "R V O V R", "O V R V O", "O R V R O"]:
        print(scheme)
        error_oracle = error_oracle_factory(DeltaF_neq, scheme)

        results[scheme] = probabilistic_bisection(error_oracle, (0, 10), n_iterations=100)

    from pickle import dump

    with open("error_thresholds.pkl", "wb") as f:
        dump((results, DeltaF_neq, sq_unc), f)
