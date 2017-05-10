# In this script, we'll take an integrator + test system, and compare various mass repartitioning schemes:
# * atoms: connected vs. all
# * mode: decrement vs. scale
# * H-mass: 1 to 4 amu


from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass, repartition_hydrogen_mass_amber
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
    n_protocol_samples, protocol_length = 100, 100
    system_name = "alanine_unconstrained"
    equilibrium_simulator = alanine_unconstrained
    target_filename = os.path.join(DATA_PATH, "hmr_{}.pkl".format(system_name))

    scheme_name, scheme = "VVVR", "O V R V O" # just use a fixed scheme for now
    timesteps = np.linspace(0.25, 2.0, 2)
    h_masses = np.linspace(1.0, 4, 2)

    hmr_modes = ["decrement", "scale"]
    hmr_atoms = ["all", "connected"]

    noneq_simulators = {}

    topology, system = equilibrium_simulator.topology, equilibrium_simulator.system

    for mode in hmr_modes:
        for atoms in hmr_atoms:
            for h_mass in h_masses:
                for timestep in timesteps:
                    hmr_system = repartition_hydrogen_mass(topology, system,
                                              h_mass=h_mass, mode=mode, atoms=atoms)
                    equilibrium_simulator.system = hmr_system
                    noneq_sim = NonequilibriumSimulator(equilibrium_simulator,
                                                        LangevinSplittingIntegrator(
                                                            splitting=scheme,
                                                            timestep=timestep * unit.femtosecond,
                                                            collision_rate=91.0 / unit.picoseconds))

                    name = "{}_{}_{}_{}".format(scheme_name, mode, atoms, timestep)
                    noneq_simulators[(name, h_mass)] = noneq_sim

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
