import benchmark.testsystems
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
#from baoab_vs_aboba_analysis import plot_results
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot, savefig

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 1000, 50
    system_name = "alanine_constrained"
    equilibrium_simulator = benchmark.testsystems.alanine_constrained
    target_filename = os.path.join(DATA_PATH, "baoab_vs_aboba_{}.pkl".format(system_name))

    schemes = {"BAOAB": "V R O O R V", "VVVR": "O V R R V O"}
    timesteps = np.linspace(0.1, 3.0, 10)
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            noneq_simulators[(name, timestep)] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(
                                                             splitting=scheme,
                                                             timestep=timestep * unit.femtosecond,
                                                             collision_rate=1.0/unit.picoseconds))
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

    with open(target_filename, "wb") as f:
        pickle.dump(results, f)

    # TODO: Plot stuff
