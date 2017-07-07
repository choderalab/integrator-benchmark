from simtk import unit
from benchmark.integrators.kyle.xchmc import XCGHMCIntegrator
import sys


steps_list = [1, 2, 3, 4, 5, 10, 50, 100]
collision_rates = [1.0 / unit.picosecond, 91.0 / unit.picosecond, None]
extra_chances_list = list(range(11))

experiments = []
for steps in steps_list:
    for collision_rate in collision_rates:
        for extra_chances in extra_chances_list:
            experiments.append((steps, collision_rate, extra_chances))


if __name__ == "__main__":
    job_id = int(sys.argv[1]) - 1

    steps, collision_rate, extra_chances = experiments[job_id]

    xchmc = XCGHMCIntegrator(steps_per_hmc=steps, timestep=3.0 * unit.femtoseconds,
                             extra_chances=extra_chances, steps_per_extra_hmc=steps,
                             collision_rate=collision_rate)

    from benchmark.testsystems import dhfr_constrained, alanine_constrained
    #testsystem = dhfr_constrained
    testsystem = alanine_constrained
    sim = testsystem.construct_simulation(xchmc)

    from mdtraj.reporters import HDF5Reporter
    reporter = HDF5Reporter(file="xchmc_steps={}, extra_chances={}, collision_rate={}.h5".format(steps, extra_chances, collision_rate), reportInterval=10, velocities=True, cell=True)
    sim.reporters.append(reporter)
    sim.runForClockTime(5 * unit.minute)
