import benchmark.testsystems
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 1000, 100
    name = "alanine_unconstrained"
    equilibrium_simulator = benchmark.testsystems.alanine_unconstrained
    target_filename = os.path.join(DATA_PATH, "baoab_vs_aboba_{}.pkl".format(name))

    schemes = {"BAOAB": "V R O R V", "ABOBA": "R V O V R"}
    noneq_simulators = {}
    for name, scheme in schemes.items():
        noneq_simulators[name] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(scheme))

    results = {}
    for marginal in ["configuration", "full"]:
        results[marginal] = {}
        for name, simulator in noneq_simulators.items():
            results[marginal][name] = simulator.collect_protocol_samples(
                n_protocol_samples, protocol_length, marginal)

    with open(target_filename, "w") as f:
        pickle.dump(results, f)