# let's sweep over the number of constrained diffusion steps, for a waterbox!


import gc
import os
import pickle

import numpy as np
from simtk import unit

from benchmark import DATA_PATH
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark.plotting import plot_scheme_comparison
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems import dhfr_constrained


def run_experiment(n_geodesic_step_list=range(1, 5), n_protocol_samples=500, protocol_length=100,
                   collision_rate="high"):
    if collision_rate == "high":
        gamma = 91.0 / unit.picosecond
    elif collision_rate == "low":
        gamma = 1.0 / unit.picosecond
    else:
        print("Defaulting to low collision_rate")
        gamma = 1.0 / unit.picosecond

    system_name = "dhfr_constrained"
    equilibrium_simulator = dhfr_constrained
    target_filename = os.path.join(DATA_PATH, "gbaoab_{}_{}_collision_rate.pkl".format(system_name, collision_rate))

    timesteps = np.linspace(1.0, 8.0, 7)
    noneq_simulators = {}
    for timestep in timesteps:
        for n_geodesic_steps in n_geodesic_step_list:
            name = "g-BAOAB ({})".format(n_geodesic_steps)
            Rs = ["R"] * n_geodesic_steps
            scheme = " ".join(["V"] + Rs + ["O"] + Rs + ["V"])
            noneq_simulators[(name, timestep)] = scheme

    results = {}
    for marginal in ["configuration", "full"]:
        results[marginal] = {}
        for ((name, timestep), scheme) in noneq_simulators.items():
            print(marginal, name, timestep)

            simulator = NonequilibriumSimulator(equilibrium_simulator,
                                                LangevinSplittingIntegrator(
                                                    splitting=scheme,
                                                    timestep=timestep * unit.femtosecond,
                                                    collision_rate=gamma))
            results[marginal][name] = simulator.collect_protocol_samples(
                n_protocol_samples, protocol_length, marginal)
            del (simulator)
            gc.collect()

            DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(*results[marginal][name])
            print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(target_filename, "wb") as f:
        pickle.dump(results, f)

    plot_scheme_comparison(target_filename, system_name)


if __name__ == "__main__":
    run_experiment(collision_rate="high")
    run_experiment(collision_rate="low")
