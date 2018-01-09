from collections import namedtuple
from pickle import dump

import numpy as np
from simtk import unit

from benchmark.evaluation import estimate_nonequilibrium_free_energy
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass_amber

ExperimentDescriptor = namedtuple("ExperimentDescriptor", ["experiment_name",
                                                           "system_name", "equilibrium_simulator",
                                                           "splitting_name", "splitting_string",
                                                           "timestep_in_fs",
                                                           "marginal",
                                                           "collision_rate_name", "collision_rate",
                                                           "n_protocol_samples", "protocol_length", "h_mass_factor"])


class Experiment():
    def __init__(self, experiment_descriptor, filename, store_potential_energy_traces=False
                 ):
        self.experiment_descriptor = experiment_descriptor
        self.filename = filename
        exp = self.experiment_descriptor
        self.store_potential_energy_traces = store_potential_energy_traces

        if exp.h_mass_factor != 1:
            if hasattr(exp.equilibrium_simulator, "MODIFIED_H_MASS"):
                raise (Exception("H mass of this system was already modified! Can't guarantee correct behavior"))

            topology, system = exp.equilibrium_simulator.topology, exp.equilibrium_simulator.system
            hmr_system = repartition_hydrogen_mass_amber(topology, system,
                                                         scale_factor=exp.h_mass_factor)
            exp.equilibrium_simulator.system = hmr_system
            exp.equilibrium_simulator.MODIFIED_H_MASS = True

    def run(self):
        exp = self.experiment_descriptor
        simulator = NonequilibriumSimulator(exp.equilibrium_simulator,
                                            LangevinSplittingIntegrator(
                                                splitting=exp.splitting_string,
                                                timestep=exp.timestep_in_fs * unit.femtosecond,
                                                collision_rate=exp.collision_rate))

        self.result = simulator.collect_protocol_samples(
            exp.n_protocol_samples, exp.protocol_length, exp.marginal,
            store_potential_energy_traces=(exp.marginal=="full" and self.store_potential_energy_traces))

        W_shads_F, W_shads_R = self.result["W_shads_F"], self.result["W_shads_R"]
        DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R)
        print(self)
        print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    def save(self):
        with open(self.filename, "wb") as f:
            everything_but_the_simulator = self.experiment_descriptor._asdict()
            everything_but_the_simulator.pop("equilibrium_simulator")

            dump({"result": self.result, "descriptor": everything_but_the_simulator}, f)

    def run_and_save(self):
        self.run()
        self.save()

    def __str__(self):
        exp = self.experiment_descriptor

        properties = [exp.system_name,
                      exp.splitting_name,
                      "dt={}fs".format(exp.timestep_in_fs),
                      "marginal: {}".format(exp.marginal),
                      "collision rate: {}".format(exp.collision_rate_name),
                      "H-mass scale: {}".format(exp.h_mass_factor)
                      ]

        return "\n\t".join(["{}"] * len(properties)).format(*properties)
