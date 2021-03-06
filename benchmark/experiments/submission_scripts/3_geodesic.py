from benchmark.testsystems import dhfr_constrained
from simtk import unit
import numpy as np
import os
from benchmark import DATA_PATH
from benchmark.experiments.driver import ExperimentDescriptor, Experiment

n_geodesic_step_list = [1,2,3,4,5]

def geodesic_ify(splitting_name, splitting_string, n_geodesic_steps=1):
    """Replace every appearance of R with several Rs"""
    Rs = " ".join(["R"]*n_geodesic_steps)

    new_name = splitting_name + " ({})".format(n_geodesic_steps)
    new_splitting_string = splitting_string.replace("R", Rs)
    return new_name, new_splitting_string

original_splittings = {"OVRVO": "O V R V O",
                      "ORVRO": "O R V R O",
                      "RVOVR": "R V O V R",
                      "VRORV": "V R O R V"
                      }

splittings = {}
for (splitting_name, splitting_string) in original_splittings.items():
    for n_geodesic_steps in n_geodesic_step_list:
        new_name, new_splitting_string = geodesic_ify(splitting_name, splitting_string, n_geodesic_steps)
        splittings[new_name] = new_splitting_string


systems = {"DHFR in explicit solvent (constrained)": dhfr_constrained}

dt_range = np.array([0.1] + list(np.arange(0.5,8.001,0.5)))

marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds}

n_protocol_samples = 100
protocol_length = 1000

experiment_name = "3_geodesic"
experiments = []
i = 0
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
