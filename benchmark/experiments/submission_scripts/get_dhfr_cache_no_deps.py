# For some reason, running into a not-yet-resolved ModuleNotFoundError when trying to use the benchmark package
# on the cluster...
# This quick script starts collecting equilibrium samples while I debug the error...

import simtk.openmm as mm
from openmmtools.integrators import GradientDescentMinimizationIntegrator, GHMCIntegrator, VVVRIntegrator
from openmmtools.testsystems import DHFRExplicit
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm

pressure = 1.0 * unit.atmosphere
temperature = 298 * unit.kelvin
n_samples = 5000
thinning_interval = 500
burn_in_length = 5000
timestep = 0.5 * unit.femtosecond

platform = mm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('CudaPrecision', 'mixed')

testsystem = DHFRExplicit(constraints=app.HBonds, rigid_water=True)
topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions
system.addForce(mm.MonteCarloBarostat(pressure, temperature))


def construct_simulation(integrator):
    """Construct a simulation instance given an integrator."""
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation


def get_positions(simulation):
    """Get array of particle positions"""
    return simulation.context.getState(getPositions=True).getPositions(asNumpy=True)


def set_positions(simulation, x):
    """Set particle positions"""
    simulation.context.setPositions(x)


def get_ghmc_acceptance_rate(simulation):
    """Return the number of acceptances divided by the number of proposals."""
    ghmc = simulation.integrator
    n_accept = ghmc.getGlobalVariableByName("naccept")
    n_trials = ghmc.getGlobalVariableByName("ntrials")
    return float(n_accept) / n_trials


def reset_ghmc_statistics(simulation):
    """Reset the number of acceptances and number of proposals."""
    ghmc = simulation.integrator
    ghmc.setGlobalVariableByName("naccept", 0)
    ghmc.setGlobalVariableByName("ntrials", 0)


def strip_unit(quantity):
    """Take a unit'd quantity and return just its value."""
    return quantity.value_in_unit(quantity.unit)


print("Collecting equilibrium samples...")
# Minimize energy by gradient descent
print("Minimizing...")
min_sim = construct_simulation(GradientDescentMinimizationIntegrator())
min_sim.context.setPositions(positions)
min_sim.context.setVelocitiesToTemperature(temperature)
for _ in tqdm(range(100)):
    min_sim.step(1)

pos = get_positions(min_sim)
del(min_sim)

# "Equilibrate" / "burn-in"
# Running a bit of Langevin first improves GHMC acceptance rates?
print("Intializing with Langevin dynamics...")
langevin_sim = construct_simulation(VVVRIntegrator(temperature=temperature, timestep=timestep))
set_positions(langevin_sim, pos)
for _ in tqdm(range(burn_in_length)):
    langevin_sim.step(1)
pos = get_positions(langevin_sim)
del(langevin_sim)

print('"Burning in" unbiased GHMC sampler for {:.3}ps...'.format(
    (burn_in_length * timestep).value_in_unit(unit.picoseconds)))
unbiased_simulation = construct_simulation(GHMCIntegrator(temperature=temperature, timestep=timestep))
set_positions(unbiased_simulation, pos)
for _ in tqdm(range(burn_in_length)):
    unbiased_simulation.step(1)
print("Burn-in GHMC acceptance rate: {:.3f}%".format(100 * get_ghmc_acceptance_rate(unbiased_simulation)))
reset_ghmc_statistics(unbiased_simulation)

# Collect equilibrium samples
print("Collecting equilibrium samples...")
unbiased_simulation.reporters.append(app.DCDReporter('dhfr_constrained_samples.dcd', thinning_interval))

for _ in tqdm(range(n_samples)):
    unbiased_simulation.step(thinning_interval)

print("Equilibrated GHMC acceptance rate: {:.3f}%".format(100 * get_ghmc_acceptance_rate(unbiased_simulation)))
