from collections import namedtuple
from pickle import dump

import numpy as np
from simtk import unit

from benchmark.evaluation import estimate_nonequilibrium_free_energy
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator

ExperimentDescriptor = namedtuple("ExperimentDescriptor", ["experiment_name",
                                                           "system_name", "equilibrium_simulator",
                                                           "splitting_name", "splitting_string",
                                                           "timestep_in_fs",
                                                           "marginal",
                                                           "collision_rate_name", "collision_rate",
                                                           "n_protocol_samples", "protocol_length", "h_mass_factor"])


class Experiment():
    def __init__(self, experiment_descriptor, filename
                 ):
        self.experiment_descriptor = experiment_descriptor
        self.filename = filename
        exp = self.experiment_descriptor

        if exp.h_mass_factor != 1:
            raise (NotImplementedError("This isn't working yet..."))

            topology, system = exp.equilibrium_simulator.topology, exp.equilibrium_simulator.system
            hmr_system = repartition_hydrogen_mass_amber(topology, system,
                                                         scale_factor=exp.h_mass_scale_factor)
            exp.equilibrium_simulator.system = hmr_system
            # TODO: Fix sequential errors... I think this will repeatedly modify the same system!
            # Need to store the original system somewhere...

    def run(self):
        exp = self.experiment_descriptor
        simulator = NonequilibriumSimulator(exp.equilibrium_simulator,
                                            LangevinSplittingIntegrator(
                                                splitting=exp.splitting_string,
                                                timestep=exp.timestep_in_fs * unit.femtosecond,
                                                collision_rate=exp.collision_rate))

        self.result = simulator.collect_protocol_samples(
            exp.n_protocol_samples, exp.protocol_length, exp.marginal)

        DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(*self.result)
        print(self)
        print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    def save(self):
        with open(self.filename, "w") as f:
            everything_but_the_simulator = self.experiment_descriptor._asdict()
            everything_but_the_simulator.pop("equilibrium_simulator")

            dump({"result": self.result, "descriptor": everything_but_the_simulator}, f)

    def run_and_save(self):
        self.run()
        self.save()

    def __str__(self):
        exp = self.experiment_descriptor

        properties = [exp.system_name, exp.splitting_name, exp.timestep_in_fs, exp.marginal,
                      exp.collision_rate_name]

        return "\n\t".join(["{}"] * len(properties)).format(*properties)
