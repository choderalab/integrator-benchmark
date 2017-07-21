from simtk import unit
from benchmark.integrators.kyle.xchmc import XCGHMCIntegrator
import sys
import numpy as np
from simtk import openmm as mm

from benchmark import simulation_parameters
#extra_chances_list = list(range(11))# + [15,20]
#extra_chances_list = list(range(20))[::-1]
extra_chances_list = [20]
#timesteps = [0.5 * unit.femtoseconds, 1.0 * unit.femtoseconds, 1.5 * unit.femtoseconds, 2.0 * unit.femtoseconds, ]
#timesteps = [1.0 * unit.femtoseconds, 2.0 * unit.femtoseconds, 3.0 * unit.femtoseconds, 4.0 * unit.femtoseconds]
#timesteps = [4.0 * unit.femtoseconds, 6.0 * unit.femtoseconds, 8.0 * unit.femtoseconds, 10.0 * unit.femtoseconds]
timesteps = [s * unit.femtoseconds for s in list(range(1,9))]
experiments = []

for timestep in timesteps:
    for extra_chances in extra_chances_list:
        experiments.append((timestep, extra_chances)) # (timestep, xtra chances)

print(len(experiments))


def evaluate(timestep, extra_chances):

    xchmc = XCGHMCIntegrator(temperature=simulation_parameters['temperature'],
                            steps_per_hmc=1, timestep=timestep,
                             extra_chances=extra_chances, steps_per_extra_hmc=1,
                             collision_rate=1.0 / unit.picosecond)

    from benchmark.testsystems import dhfr_constrained, waterbox_constrained, alanine_unconstrained, alanine_constrained
    #testsystem = dhfr_constrained
    testsystem = waterbox_constrained
    #testsystem = alanine_constrained
    #platform = mm.Platform.getPlatformByName("Reference")
    #platform = mm.Platform.getPlatformByName('CUDA')
    #platform.setPropertyDefaultValue('CudaPrecision', 'mixed') # double precision now...
    platform = mm.Platform.getPlatformByName('OpenCL')
    platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
    #platform.setPropertyDefaultValue('DeterministicForces', 'true')
    testsystem.platform = platform
    sim = testsystem.construct_simulation(xchmc)


    # pick an equilibrated initial condition
    testsystem.load_or_simulate_x_samples() # make sure x_samples is there...

    sim.context.setPositions(testsystem.x_samples[-1])

    from mdtraj.reporters import HDF5Reporter
    name = "xchmc_timestep={}fs, extra_chances={}".format(timestep.value_in_unit(unit.femtoseconds), extra_chances)
    reporter = HDF5Reporter(file=name + ".h5", reportInterval=1000, velocities=True, cell=True)
    #sim.reporters.append(reporter)

    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        #sim.runForClockTime(10 * unit.minute)
        sim.context.setPositions(testsystem.sample_x_from_equilibrium())
        sim.context.setVelocitiesToTemperature(simulation_parameters['temperature'])
        sim.step(1)

    print(xchmc.all_counts)
    np.save(name + ".npy", xchmc.all_counts)

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     job_id = int(sys.argv[1]) - 1
    #     timestep, extra_chances = experiments[job_id]
    # else:
    #     timestep, extra_chances = 0.1 * unit.femtoseconds, 20
    #
    # evaluate(timestep, extra_chances)

    for (timestep, extra_chances) in experiments:
        evaluate(timestep, extra_chances)