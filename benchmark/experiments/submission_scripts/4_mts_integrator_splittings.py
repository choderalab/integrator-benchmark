import os

import numpy as np
from simtk import unit

from benchmark import DATA_PATH
from benchmark.experiments.driver import ExperimentDescriptor, Experiment
from benchmark.testsystems import dhfr_constrained

n_inner_step_list = [1, 2, 3, 4, 5]


def simplify_mts_string(mts_string):
    simplified_string = mts_string.replace("V1 V1", "V1")
    return simplified_string


def generate_baoab_mts_string(n_inner_steps=1):
    full_string = "V0 " + " ".join(["V1 R O R V1"] * n_inner_steps) + " V0"

    return simplify_mts_string(full_string)


def generate_standard_mts_string(n_inner_steps=1):
    full_string = "O V0 " + " ".join(["V1 R V1"] * n_inner_steps) + " V0 O"
    return simplify_mts_string(full_string)


splittings = {}
for n in n_inner_step_list:
    splittings["BAOAB MTS ({})".format(n)] = generate_baoab_mts_string(n)
    splittings["Standard MTS ({})".format(n)] = generate_standard_mts_string(n)
    print("{}\n\t{}\n\t{}".format(n, generate_baoab_mts_string(n), generate_standard_mts_string(n)))

# modify system forces to put nonbonded terms in fg0, valence terms in fg1
forces = dhfr_constrained.system.getForces()

for force in forces:
    if "Nonbonded" in force.__class__.__name__:
        force.setForceGroup(0)
    else:
        force.setForceGroup(1)

systems = {"DHFR in explicit solvent (constrained)": dhfr_constrained}

dt_range = np.array([0.1] + list(range(1, 17)))

marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds}

n_protocol_samples = 100
protocol_length = 2000

experiment_name = "5_mts_integrator_splittings"
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
                        n_protocol_samples=n_protocol_samples,
                        protocol_length=protocol_length,
                        h_mass_factor=1
                    )

                    experiments.append(Experiment(experiment_descriptor, full_filename))
                    i += 1

print(len(experiments))

if __name__ == "__main__":
    import sys

    job_id = int(sys.argv[1])
    experiments[job_id].run_and_save()
