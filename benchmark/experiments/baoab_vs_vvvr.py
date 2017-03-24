import benchmark.testsystems
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.plotting import plot_scheme_comparison
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 1000, 50
    system_name = "alanine_unconstrained"
    equilibrium_simulator = benchmark.testsystems.alanine_unconstrained
    target_filename = os.path.join(DATA_PATH, "scheme_comparison_{}.pkl".format(system_name))

    schemes = {"BAO": "V R O", "BABO": "V R V O",
               "ABO": "R V O", "ABAO": "R V R O"}
    timesteps = np.linspace(0.1, 1.5, 5)
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            noneq_simulators[(name, timestep)] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(
                                                             splitting=scheme, timestep=timestep * unit.femtosecond))
    # need to catch "Exception: Particle coordinate is nan"

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
            print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(target_filename, "w") as f:
        pickle.dump(results, f)

    plot_scheme_comparison(target_filename, system_name)