# In this experiment, we see how the estimated configuration-space KL divergence depends on the protocol length
# for a fixed choice of integrator and timestep

import benchmark.testsystems
from benchmark.testsystems import NumbaNonequilibriumSimulator
from benchmark import DATA_PATH
import pickle
import numpy as np

from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.integrators import vvvr_factory
from functools import partial

from benchmark import FIGURE_PATH
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    n_protocol_samples = 1000000
    system_name = "quartic"
    equilibrium_simulator = benchmark.testsystems.quartic
    experiment_name = "protocol_length_test_{}".format(system_name)
    data_filename = os.path.join(DATA_PATH, "{}.pkl".format(experiment_name))
    figure_filename = os.path.join(FIGURE_PATH, "{}.jpg".format(experiment_name))

    potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
    timestep = 1.1
    gamma = 100.0

    noneq_simulator = NumbaNonequilibriumSimulator(equilibrium_simulator, partial(vvvr_factory(potential, force, velocity_scale, mass), gamma=gamma, dt=timestep))
    results = {}
    protocol_lengths = range(1,10) + range(10, 100)
    means = []
    errors = []
    for protocol_length in protocol_lengths:
        W_shads_F, W_shads_R = noneq_simulator.collect_protocol_samples(
            n_protocol_samples, protocol_length, "configuration")
        results[protocol_length] = W_shads_F, W_shads_R

        W_shads_F = W_shads_F
        W_shads_R = W_shads_R
        DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R)

        means.append(DeltaF_neq)
        errors.append(np.sqrt(squared_uncertainty))
        print("\t{:.5f} +/- {:.5f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(data_filename, "w") as f:
        pickle.dump(results, f)

    plt.figure()
    plt.errorbar(protocol_lengths, means, yerr=errors)
    plt.xlabel("Protocol length (# steps)")
    plt.ylabel("Estimated $\Delta F_{neq}$")
    plt.title("Dependence of $\Delta F_{neq}$ estimate on protocol length")
    plt.savefig(os.path.join(FIGURE_PATH, figure_filename))
