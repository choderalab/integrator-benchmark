from simtk import unit
from benchmark.integrators.kyle.xchmc import XCGHMCIntegrator
import numpy as np
from simtk import openmm as mm

from benchmark import simulation_parameters

timestep = 0.5 * unit.femtoseconds


def evaluate(timestep, extra_chances):

    xchmc = XCGHMCIntegrator(temperature=simulation_parameters['temperature'],
                            steps_per_hmc=1, timestep=timestep,
                             extra_chances=extra_chances, steps_per_extra_hmc=1,
                             collision_rate=1.0 / unit.picosecond)

    from benchmark.testsystems import dhfr_constrained
    testsystem = dhfr_constrained

    platform = mm.Platform.getPlatformByName('CUDA')
    platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
    platform.setPropertyDefaultValue('DeterministicForces', 'true')
    testsystem.platform = platform
    sim = testsystem.construct_simulation(xchmc)

    # pick an equilibrated initial condition
    testsystem.load_or_simulate_x_samples() # make sure x_samples is there...
    sim.context.setPositions(testsystem.x_samples[-1])

    from mdtraj.reporters import HDF5Reporter
    name = "dhfr_xchmc_timestep={}fs, extra_chances={}".format(timestep.value_in_unit(unit.femtoseconds), extra_chances)
    reporter = HDF5Reporter(file=name + ".h5", reportInterval=1000, velocities=True, cell=True)
    sim.reporters.append(reporter)

    from tqdm import tqdm

    for _ in tqdm(range(10000)):
        sim.context.setPositions(testsystem.sample_x_from_equilibrium())
        sim.context.setVelocitiesToTemperature(simulation_parameters['temperature'])
        sim.step(1)

        print(xchmc.all_counts)
    np.save(name + ".npy", xchmc.all_counts)

if __name__ == "__main__":
    evaluate(timestep=0.5 * unit.femtoseconds,
             extra_chances=20)