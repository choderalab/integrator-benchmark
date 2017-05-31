import os

import numpy as np
from simtk import unit

from benchmark import DATA_PATH
from benchmark.experiments.driver import ExperimentDescriptor, Experiment
from benchmark.testsystems import dhfr_constrained, alanine_constrained, t4_constrained, waterbox_constrained

splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }

systems = {"Alanine dipeptide in vacuum (constrained)": alanine_constrained,
           "T4 lysozyme in implicit solvent (constrained)": t4_constrained,
           "DHFR in explicit solvent (constrained)": dhfr_constrained,
           "TIP3P water (rigid)": waterbox_constrained
           }

dt_range = np.array([0.1] + list(np.arange(0.5, 10.001, 0.5)))

marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds}

n_protocol_samples = {"Alanine dipeptide in vacuum (constrained)": 10000,
                      "T4 lysozyme in implicit solvent (constrained)": 1000,
                      "DHFR in explicit solvent (constrained)": 1000,
                      "TIP3P water (rigid)": 1000
                      }

protocol_length = 1000

experiment_name = "2_systems"
experiments = []
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
                        n_protocol_samples=n_protocol_samples[system_name],
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
