# let's sweep over the number of constrained diffusion steps, for a waterbox!


from benchmark.testsystems import NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
from benchmark.testsystems import dhfr_constrained

import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import plot_scheme_comparison

def run_experiment(n_geodesic_step_list=range(1,5), n_protocol_samples=500, protocol_length=100, friction="high"):
    if friction == "high":
        collision_rate = 91.0 / unit.picosecond
    elif friction == "low":
        collision_rate = 1.0 / unit.picosecond
    system_name = "dhfr_constrained"
    equilibrium_simulator = dhfr_constrained
    target_filename = os.path.join(DATA_PATH, "gbaoab_{}_friction{}.pkl".format(friction, system_name))

    timesteps = np.linspace(0.5, 4.5, 5)
    noneq_simulators = {}
    for timestep in timesteps:
        for n_geodesic_steps in n_geodesic_step_list:
            name = "g-BAOAB ({})".format(n_geodesic_steps)
            Rs = ["R"] * n_geodesic_steps
            scheme = " ".join(["V"] + Rs + ["O"] + Rs + ["V"])
            noneq_simulators[(name, timestep)] = NonequilibriumSimulator(equilibrium_simulator,
                                                         LangevinSplittingIntegrator(
                                                             splitting=scheme,
                                                             timestep=timestep * unit.femtosecond,
                                                             collision_rate=collision_rate))
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


if __name__ == "__main__":
    run_experiment(friction="high")
    run_experiment(friction="low")
