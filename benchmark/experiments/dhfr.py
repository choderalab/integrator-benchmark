from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH

import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot_scheme_comparison
from benchmark.testsystems import dhfr_constrained

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 100, 1000
    system_name = "dhfr_constrained"
    equilibrium_simulator = dhfr_constrained
    target_filename = os.path.join(DATA_PATH, "dhfr_{}.pkl".format(system_name))

    schemes = {"g-BAOAB (1)": "V R O R V",
               "g-BAOAB (2)": "V R R O R R V",
               "g-BAOAB (3)": "V R R R O R R R V",
               "g-BAOAB (4)": "V R R R R O R R R R V",
               "VVVR": "O V R V O"
               }
    timesteps = np.linspace(0.1, 2.0, 5)
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            noneq_simulators[(name, timestep)] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(
                                                             splitting=scheme,
                                                             timestep=timestep * unit.femtosecond,
                                                             collision_rate=91.0/unit.picoseconds))
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
