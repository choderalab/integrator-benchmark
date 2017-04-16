# In this experiment, we validate the nonequilibrium estimator of configuration-marginal KL divergence
# on three schemes: ABOBA, BAOAB, and OBABO


# 1. Collect samples from steady state for each scheme (exact, BAOAB, ABOBA, OBABO)
# 2. Estimate KL divergences using "ground-truth" numerical methods (using histograms)
# 3. Estimate KL divergences using nonequilibrium scheme

# collect steady state samples


import benchmark.testsystems
from benchmark.testsystems import NumbaNonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.plotting import plot_scheme_comparison
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.integrators import baoab_factory, vvvr_factory, aboba_factory
from functools import partial


if __name__ == "__main__":
    n_protocol_samples, protocol_length = 2000000, 10
    system_name = "quartic"
    equilibrium_simulator = benchmark.testsystems.quartic
    target_filename = os.path.join(DATA_PATH, "scheme_comparison_{}.pkl".format(system_name))

    potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
    schemes = {"BAOAB": baoab_factory(potential, force, velocity_scale, mass),
               "VVVR": vvvr_factory(potential, force, velocity_scale, mass),
               "ABOBA": aboba_factory(potential, force, velocity_scale, mass),
               }
    timesteps = np.linspace(0.1, 1.1, 10)
    gamma = 100.0
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            # output of the integrator factory is a function with this signature: simulate_aboba(x0, v0, n_steps, gamma, dt)
            #integrator = lambda x_0, v_0, n_steps : scheme(x_0, v_0, n_steps, gamma, timestep)
            noneq_simulators[(name, timestep)] = NumbaNonequilibriumSimulator(equilibrium_simulator,
                                                                              partial(scheme, gamma=gamma, dt=timestep))

    results = {}
    for marginal in ["configuration", "full"]:
        results[marginal] = {}
        for name, simulator in noneq_simulators.items():
            print(marginal, name)
            W_shads_F, W_shads_R = simulator.collect_protocol_samples(
                n_protocol_samples, protocol_length, marginal)
            results[marginal][name] = W_shads_F, W_shads_R

            W_shads_F = W_shads_F
            W_shads_R = W_shads_R
            DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R)
            print("\t{:.5f} +/- {:.5f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(target_filename, "wb") as f:
        pickle.dump(results, f)

    plot_scheme_comparison(target_filename, system_name)
