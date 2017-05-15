# let's sweep over the number of constrained diffusion steps, for a waterbox!


from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
from benchmark.testsystems import waterbox_constrained, constraint_coupled_harmonic_oscillators, t4_constrained

import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot_scheme_comparison

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 500, 100
    #system_name = "constraint_coupled_harmonic_oscillators"
    #equilibrium_simulator = constraint_coupled_harmonic_oscillators
    #system_name = "waterbox_constrained"
    #equilibrium_simulator = waterbox_constrained
    system_name = "t4_constrained"
    equilibrium_simulator = t4_constrained
    target_filename = os.path.join(DATA_PATH, "gbaoab_{}.pkl".format(system_name))

    schemes = {"g-BAOAB (1)": "V R O R V",
               "g-BAOAB (2)": "V R R O R R V",
               "g-BAOAB (3)": "V R R R O R R R V",
               "g-BAOAB (4)": "V R R R R O R R R R V"}
    timesteps = np.linspace(1.0, 5.0, 5)
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