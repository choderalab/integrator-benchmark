import benchmark.testsystems
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot_scheme_comparison

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 10000, 100
    system_name = "alanine_unconstrained_low_friction"
    equilibrium_simulator = benchmark.testsystems.alanine_unconstrained
    target_filename = os.path.join(DATA_PATH, "baoab_vs_vvvr_{}.pkl".format(system_name))

    schemes = {"BAOAB": "V R O R V", "VVVR": "O V R V O"}
    timesteps = np.linspace(0.1, 3, 10)
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            noneq_simulators[(name, timestep)] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(
                                                             splitting=scheme,
                                                             timestep=timestep * unit.femtosecond,
                                                             collision_rate=1.0/unit.picoseconds))
    results = {}
    for marginal in ["configuration", "full"]:
        results[marginal] = {}
        for name, simulator in noneq_simulators.items():
            print(marginal, name)
            results[marginal][name] = simulator.collect_protocol_samples(
                n_protocol_samples, protocol_length, marginal)

            DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(*results[marginal][name])
            print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(target_filename, "wb") as f:
        pickle.dump(results, f)

    plot_scheme_comparison(target_filename, system_name)
