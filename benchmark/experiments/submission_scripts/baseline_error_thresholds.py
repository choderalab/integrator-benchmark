# For each system, compute the configuration-space error to high accuracy.

import os

from simtk import unit

from benchmark import DATA_PATH
from benchmark.experiments.driver import Experiment, ExperimentDescriptor
from benchmark.testsystems import dhfr_constrained, dhfr_unconstrained, \
    waterbox_constrained, flexible_waterbox, t4_constrained, t4_unconstrained, \
    alanine_constrained, alanine_unconstrained, src_constrained

systems = {"DHFR in explicit solvent (constrained)": dhfr_constrained,
           "DHFR in explicit solvent (unconstrained)": dhfr_unconstrained,
           "Src in explicit solvent (constrained)": src_constrained,
           "Waterbox (constrained)": waterbox_constrained,
           "Waterbox (unconstrained)": flexible_waterbox,
           "T4 lysozyme in implicit solvent (constrained)": t4_constrained,
           "T4 lysozyme in implicit solvent (unconstrained)": t4_unconstrained,
           "Alanine dipeptide in vacuum (constrained)": alanine_constrained,
           "Alanine dipeptide in vacuum (unconstrained)": alanine_unconstrained}

keys = sorted(systems.keys())
for i in range(len(keys)):
    print(i + 1, keys[i])

n_protocol_samples = 1000
protocol_length = 2000

splitting_name = "OVRVO"
splitting = "O V R V O"

marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds}

experiment_name = "0_baseline"
experiments = []
i = 1
for system_name in keys:
    partial_fname = "{}_{}.pkl".format(experiment_name, i)
    full_filename = os.path.join(DATA_PATH, partial_fname)

    experiment_descriptor = ExperimentDescriptor(
        experiment_name=experiment_name,
        system_name=system_name,
        equilibrium_simulator=systems[system_name],
        splitting_name=splitting_name,
        splitting_string=splitting,
        timestep_in_fs=2.0 * unit.femtosecond,
        marginal="configuration",
        collision_rate_name="low",
        collision_rate=collision_rates["low"],
        n_protocol_samples=n_protocol_samples,
        protocol_length=protocol_length,
        h_mass_factor=1
    )

    experiments.append(Experiment(experiment_descriptor, full_filename))
    i += 1

if __name__ == "__main__":
    print(len(experiments))
    import sys

    job_id = int(sys.argv[1])
    experiments[job_id].run_and_save()
