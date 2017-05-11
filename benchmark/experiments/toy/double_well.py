#
import benchmark.testsystems
from benchmark.testsystems import NumbaNonequilibriumSimulator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from benchmark.plotting import plot_scheme_comparison
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.integrators import baoab_factory, vvvr_factory, aboba_factory
from functools import partial

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 10000000, 10
    system_name = "double_well"
    equilibrium_simulator = benchmark.testsystems.double_well
    target_filename = os.path.join(DATA_PATH, "scheme_comparison_{}.pkl".format(system_name))

    potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
    schemes = {"VRORV": baoab_factory(potential, force, velocity_scale, mass),
               "OVRVO": vvvr_factory(potential, force, velocity_scale, mass),
               "RVOVR": aboba_factory(potential, force, velocity_scale, mass),
               }
    timesteps = np.linspace(0.1, 0.6, 4)
    gamma = 100.0
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
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
