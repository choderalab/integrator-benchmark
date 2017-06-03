import os
import sys

from simtk import unit

from benchmark import DATA_PATH
from benchmark.experiments.driver import ExperimentDescriptor, Experiment
from benchmark.testsystems import dhfr_constrained
import numpy as np

scale_factors = np.arange(1.0, 4.01, 0.25)
dt_range = np.arange(0.5, 6.01, 0.5)

splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }

system_name = "DHFR in explicit solvent (constrained)"
system = dhfr_constrained

marginals = ["configuration", "full"]

collision_rate_name = "low"
collision_rate = 1.0 / unit.picoseconds

n_protocol_samples = 1000
protocol_length = 2000

experiment_name = "5_hmr"
descriptors_and_fnames = []

i = 1
for scale_factor in scale_factors[::-1]:  # start from most interesting
    for splitting_name in sorted(splittings.keys()):
        for dt in dt_range[::-1]:  # start from most interesting
            for marginal in marginals:
                partial_fname = "{}_{}.pkl".format(experiment_name, i)
                full_filename = os.path.join(DATA_PATH, partial_fname)

                experiment_descriptor = ExperimentDescriptor(
                    experiment_name=experiment_name,
                    system_name=system_name,
                    equilibrium_simulator=system,
                    splitting_name=splitting_name,
                    splitting_string=splittings[splitting_name],
                    timestep_in_fs=dt,
                    marginal=marginal,
                    collision_rate_name=collision_rate_name,
                    collision_rate=collision_rate,
                    n_protocol_samples=n_protocol_samples,
                    protocol_length=protocol_length,
                    h_mass_factor=scale_factor
                )

                descriptors_and_fnames.append((experiment_descriptor, full_filename))
                i += 1

print(len(descriptors_and_fnames))

if __name__ == "__main__":
    job_id = int(sys.argv[1])
    experiment_descriptor, full_filename = descriptors_and_fnames[job_id]
    Experiment(experiment_descriptor, full_filename, store_potential_energy_traces=True).run_and_save()
