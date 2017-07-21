# Different integrator splittings have different maximum tolerated timesteps that achieve the same error threshold

import os

import numpy as np
from simtk import unit

from benchmark import DATA_PATH
from benchmark.testsystems import dhfr_constrained
from benchmark.experiments.driver import ExperimentDescriptor, Experiment

splittings = {"RVOVR": "R V O V R",
              "RVOOVR": "R V O O V R",
              }

systems = {"DHFR (constrained)": dhfr_constrained}

dt_range = np.array([3.5])

marginals = ["configuration"]

collision_rates = {"low": 1.0 / unit.picoseconds,
                   "high": 91.0 / unit.picoseconds}

n_protocol_samples = 100
protocol_length = 4000

experiment_name = "A3_extra_O_steps"
experiments = [None]
i = 1
for splitting_name in sorted(splittings.keys()):
    for system_name in sorted(systems.keys()):
        for dt in dt_range:
            for marginal in marginals:
                for collision_rate_name in sorted(collision_rates.keys()):
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
