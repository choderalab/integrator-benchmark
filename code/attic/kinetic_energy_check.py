import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from simtk import unit
from simtk import openmm as mm
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from openmmtools.integrators import GHMCIntegrator
import numpy as np
temperature = 298.0 * unit.kelvin
ke_unit = unit.kilojoule_per_mole

# masses of all particles in the water box
system = WaterBox().system
m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])

def get_velocities(sim):
    return sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)

def get_official_ke(sim):
    """Computes kinetic energy after projecting current
    velocities onto constraint manifold, as in:
    https://github.com/pandegroup/openmm/blob/master/platforms/reference/src/ReferenceKernels.cpp#L164-L194
    """
    return sim.context.getState(getEnergy=True).getKineticEnergy()

def get_uncorrected_ke(sim):
    """Compute KE as 1/2 m v^2, regardless if v has any component not tangent
    to the velocity constraint surface."""
    v2 = (sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)**2).sum(1)
    return np.sum(0.5 * m * v2)

def strip_unit(quantity):
    try: return quantity.value_in_unit(quantity.unit)
    except: return quantity

def get_ke(sim):
    """Return a tuple of the "official" KE, and the 0.5mv^2 KE."""
    ke_0 = strip_unit(get_official_ke(sim))
    ke_1 = strip_unit(get_uncorrected_ke(sim))
    return ke_0, ke_1

def sample_kinetic_energy(sim, temperature):
    """Sample a velocity vector from (constrained) Maxwell-Boltzmann distribution,
    report the kinetic energy."""
    sim.context.setVelocitiesToTemperature(temperature)
    return get_ke(sim)

# Check if the distribution of kinetic energies for
# `v_0 = setVelocitiesToTemperature() + constrain(sigma * gaussian)`
# is equal to the distribution of kinetic energies for
# `v_1 = constrain(setVelocitiesToTemperature() + sigma * gaussian)`
# (where `sigma = sqrt(kT / m)`.
def sample_kinetic_energy_0(sim, temperature):
    """Method 0:
    (1) Draw v, v' i.i.d. from constrained Maxwell-Boltzmann
    (2) Return v + v'

    (What I think we should be doing)
    """
    sim.context.setVelocitiesToTemperature(temperature)
    v = get_velocities(sim)
    sim.context.setVelocitiesToTemperature(temperature)
    v_ = get_velocities(sim)
    sim.context.setVelocities(v + v_)
    return get_ke(sim)

def sample_kinetic_energy_1(sim, unconstrained_sim, temperature):
    """Method 1:
    (1) Draw v from constrained Maxwell-Boltzmann
    (2) Draw u from *unconstrained* Maxwell-Boltzmann
    (3) Set velocity to be u + v
    (4) Apply velocity constraints
    (5) Return velocity

    (What we currently are doing)
    """
    sim.context.setVelocitiesToTemperature(temperature)
    v = get_velocities(sim)
    # draw from unconstrained Maxwell-Boltzmann
    unconstrained_sim.setVelocitiesToTemperature(temperature)
    u = get_velocities(unconstrained_sim)
    sim.context.setVelocities(u + v)

    # apply velocity constraints to sim

    return get_velocities(sim)


def create_sim(integrator, constrained=True):
    """Create Simulation object for WaterBox"""
    testsystem = WaterBox(constrained=constrained)
    (system, positions) = testsystem.system, testsystem.positions
    sim = app.Simulation(testsystem.topology, system, integrator)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(temperature)
    return sim

def get_kinetic_energy_distribution(n_samples):
    """Draw velocities and report KEs for velocities drawn independently from
    Maxwell-Boltzmann distribution."""
    constrained_sim = create_sim(GHMCIntegrator(temperature), constrained=True)
    unconstrained_sim = create_sim(GHMCIntegrator(temperature), constrained=False)

    unconstrained_kes = []
    constrained_kes = []
    for _ in tqdm(range(n_samples)):
        constrained_kes.append(sample_kinetic_energy(constrained_sim, temperature))
        unconstrained_kes.append(sample_kinetic_energy(unconstrained_sim, temperature))
    return unconstrained_kes, constrained_kes

def get_kinetic_energy_distribution_simulation(n_samples, integrator='GHMC', burnin=100):
    """Simulate either GHMC or Langevin dynamics, and report the observed KE marginal."""
    if integrator == "GHMC":
        integrator_factory = lambda : GHMCIntegrator(temperature)
    else:
        def integrator_factory():
            temp = temperature.value_in_unit(unit.kelvin)
            frictionCoeff = 91 # inverse picoseconds
            stepSize = (1*unit.femtosecond).value_in_unit(unit.picosecond)
            integrator = mm.LangevinIntegrator(temp, frictionCoeff, stepSize)
            return integrator

    constrained_sim = create_sim(integrator_factory(), constrained=True)
    unconstrained_sim = create_sim(integrator_factory(), constrained=False)

    constrained_sim.step(burnin)
    unconstrained_sim.step(burnin)

    unconstrained_kes = []
    constrained_kes = []

    for _ in tqdm(range(n_samples)):
        constrained_sim.step(1)
        unconstrained_sim.step(1)
        ke_c = get_ke(constrained_sim)
        ke_u = get_ke(unconstrained_sim)
        constrained_kes.append(ke_c)
        unconstrained_kes.append(ke_u)
    return unconstrained_kes, constrained_kes


def plot_results(results, name=""):
    """Given a list of (label, data) results, plot histograms
    on the same axis and save result."""
    style = {"bins": 50,
             "histtype":"stepfilled",
             "normed":True,
             "alpha":0.5
             }

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (label, data) in results:
        ax.hist(data, label=label, **style)

    lgd = ax.legend(loc=(1,0))
    ax.set_xlabel('Kinetic energy ({})'.format(str(ke_unit)))
    ax.set_ylabel('Frequency')
    ax.set_title('WaterBox kinetic energy distribution')
    fig.savefig('kinetic_energy_check_{}.jpg'.format(name), dpi=300,
                bbox_extra_artists = (lgd,), bbox_inches = 'tight')
    plt.close()

n_samples = 10000
langevin = get_kinetic_energy_distribution_simulation(n_samples, integrator="Langevin")
ghmc = get_kinetic_energy_distribution_simulation(n_samples,integrator="GHMC")
independent = get_kinetic_energy_distribution(n_samples)

results = []
sim_schemes = {"Langevin": langevin, "GHMC": ghmc, "Independent": independent}
constraint_modes = ["Unconstrained", "Constrained"]
ke_measurement_types = [".getKineticEnergy()", "mv^2"]
for sim_scheme in sim_schemes.keys():
    for constrained in range(2):
        for ke_measurement_type in range(2):
            label = "{} {} ({})".format(constraint_modes[constrained], sim_scheme, ke_measurement_types[ke_measurement_type])
            data = [e[ke_measurement_type] for e in sim_schemes[sim_scheme][constrained]]
            results.append((label, data))
            print(label, np.mean(data))

plot_results(results, "full_comparison")