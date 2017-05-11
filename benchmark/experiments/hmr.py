# In this script, we'll take an integrator + test system, and compare various mass repartitioning schemes:
# * atoms: connected vs. all
# * mode: decrement vs. scale
# * H-mass: 1 to 4 amu


from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass_amber
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot_scheme_comparison
from benchmark.testsystems import alanine_unconstrained

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 5000, 50
    system_name = "alanine_unconstrained"
    equilibrium_simulator = alanine_unconstrained
    target_filename = os.path.join(DATA_PATH, "hmr_{}.pkl".format(system_name))

    scheme_name, scheme = "VVVR", "O V R V O" # just use a fixed scheme for now
    timesteps = np.linspace(0.5, 3.0, 10)
    scale_factors = [1, 1.5, 2, 2.5, 3, 3.5, 4]

    noneq_simulators = {}

    topology, system = equilibrium_simulator.topology, equilibrium_simulator.system


    for scale_factor in scale_factors:
        for timestep in timesteps:
            hmr_system = repartition_hydrogen_mass_amber(topology, system,
                                      scale_factor=scale_factor)
            equilibrium_simulator.system = hmr_system
            noneq_sim = NonequilibriumSimulator(equilibrium_simulator,
                                                LangevinSplittingIntegrator(
                                                    splitting=scheme,
                                                    timestep=timestep * unit.femtosecond,
                                                    collision_rate=91.0 / unit.picoseconds))

            name = "{}_{}".format(scheme_name, scale_factor)
            noneq_simulators[(name, timestep)] = noneq_sim

    results = {}
    for marginal in ["configuration"]:#, "full"]: # just looking at configuration properties for now
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
