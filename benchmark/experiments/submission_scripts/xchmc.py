from simtk import unit
from benchmark.integrators.kyle.xchmc import XCGHMCIntegrator
import sys
import numpy as np

extra_chances_list = list(range(11))
timesteps = [0.5 * unit.femtoseconds, 1.0 * unit.femtoseconds, 1.5 * unit.femtoseconds, 2.0 * unit.femtoseconds]

experiments = []

for timestep in timesteps:
    for extra_chances in extra_chances_list:
        experiments.append((timestep, extra_chances)) # (timestep, xtra chances)

if __name__ == "__main__":
    job_id = int(sys.argv[1]) - 1

    timestep, extra_chances = experiments[job_id]

    xchmc = XCGHMCIntegrator(steps_per_hmc=1, timestep=timestep,
                             extra_chances=extra_chances, steps_per_extra_hmc=1,
                             collision_rate=1.0 / unit.picosecond)

    from benchmark.testsystems import dhfr_constrained
    testsystem = dhfr_constrained
    sim = testsystem.construct_simulation(xchmc)

    from mdtraj.reporters import HDF5Reporter
    name = "xchmc_timestep={}fs, extra_chances={}".format(timestep.value_in_unit(unit.femtoseconds), extra_chances)
    reporter = HDF5Reporter(file=name + ".h5", reportInterval=100, velocities=True, cell=True)
    sim.reporters.append(reporter)
    sim.runForClockTime(10 * unit.minute)

    np.save(name + ".npy", xchmc.all_counts)
