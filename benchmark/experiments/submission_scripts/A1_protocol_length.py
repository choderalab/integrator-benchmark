# Examine how the nonequilibrium estimate depends on protocol length

import os

import numpy as np
from simtk import unit

from benchmark import DATA_PATH
from benchmark.experiments.driver import ExperimentDescriptor, Experiment
from benchmark.testsystems import dhfr_constrained

splittings = {"VRORV": "V R O R V"}

systems = {"DHFR (constrained)": dhfr_constrained}

dt_range = np.array([2.0, 4.0])

marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds,
                   "high": 91.0 / unit.picoseconds}

n_protocol_samples = 1000
protocol_lengths = [1, 2, 3, 4, 5, 10, 20, 50, 100, 250, 500, 1000, 2000]

experiment_name = "A1_protocol_length"
experiments = []
i = 1
for splitting_name in sorted(splittings.keys()):
    for system_name in sorted(systems.keys()):
        for dt in dt_range:
            for marginal in marginals:
                for collision_rate_name in sorted(collision_rates.keys()):
                    for protocol_length in protocol_lengths:
                        partial_fname = "{}_{}.pkl".format(experiment_name, i)
                        full_filename = os.path.join(DATA_PATH, partial_fname)

                        experiment_descriptor = ExperimentDescriptor(
                            experiment_name=experiment_name,
                            system_name=system_name,
                            equilibrium_simulator=systems[system_name],
                            splitting_name=splitting_name,
                            splitting_string=splittings[splitting_name],
                            timestep_in_fs=dt,
                            marginal=marginal,
                            collision_rate_name=collision_rate_name,
                            collision_rate=collision_rates[collision_rate_name],
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
