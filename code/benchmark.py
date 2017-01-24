""""""

from simtk import unit
from simtk.openmm import app
import numpy as np
from integrators import LangevinSplittingIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
W_unit = unit.kilojoule_per_mole
from analysis import estimate_nonequilibrium_free_energy
from cPickle import dump

from openmmtools.integrators import GHMCIntegrator

from utils import plot, get_total_energy, get_summary_string, configure_platform

def measure_shadow_work(simulation, n_steps):
    """Simulate for n_steps, and record the integrator's W_shad global variable
    at each step, minus the value of W_shad before integrating."""
    get_W_shad = lambda : simulation.integrator.getGlobalVariableByName("W_shad")
    W_shads = np.zeros(n_steps)
    init_W_shad = get_W_shad()
    for i in range(n_steps):
        simulation.step(1)
        W_shads[i] = get_W_shad()
    return W_shads - init_W_shad

def measure_shadow_work_via_heat(simulation, n_steps):
    """Given a `simulation` that uses an integrator that accumulates heat exchange with bath,
    apply the integrator for n_steps and return the change in energy - the heat."""
    get_energy = lambda : get_total_energy(simulation)
    get_heat = lambda : simulation.integrator.getGlobalVariableByName("heat")

    E_0 = get_energy()
    Q_0 = get_heat()

    W_shads = []

    for _ in range(n_steps):
        simulation.step(1)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        W_shad = delta_E.value_in_unit(W_unit) - delta_Q
        W_shads.append(W_shad)

    return np.array(W_shads)

def randomization_midpoint_operator(simulation, temperature):
    """Resamples velocities from Maxwell-Boltzmann distribution."""
    simulation.context.setVelocitiesToTemperature(temperature)

def null_midpoint_operator(simulation):
    """Do nothing to the simulation"""
    pass

def benchmark(simulation, equilibrium_samples, n_samples, M, midpoint_operator, temperature):
    """Estimate the nonequilbrium free energy difference between the equilibrium ensemble and
    the perturbed ensemble sampled by the simulation.

    For each of n_samples:
    * Draw a configuration sample with replacement from equilibrium_samples,
    * Draw a velocity from the Maxwell-Boltzmann distribution
    * Simulate for M steps, accumulating shadow work
    * Apply the midpoint operator (either does nothing, or randomizes velocities)
    * Simulate for M steps, accumulating shadow work

    Return:
        * W_shads_F: work trajectories from applying Langevin integrator to unbiased samples
        * W_shads_R: work trajectories from applying Langevin integrator to samples prepared
        in nonequilibrium ensemble
        * DeltaF_neq: estimate of nonequilibrium free energy difference
        * sq_uncertainty: estimate of squared uncertainty in estimated nonequilibrium free
        energy difference
        * W_midpoint: work values for applying midpoint operator to nonequilibrium samples
    """
    W_shads_F, W_shads_R, W_midpoint = [], [], []

    for _ in range(n_samples):
        # draw equilibrium sample
        simulation.context.setPositions(equilibrium_samples[np.random.randint(len(equilibrium_samples))])
        simulation.context.setVelocitiesToTemperature(temperature)

        # perform "forward" protocol
        W_shads_F.append(measure_shadow_work(simulation, M))

        # perform midpoint operation
        E_before_midpoint = get_total_energy(simulation)
        midpoint_operator(simulation)
        E_after_midpoint = get_total_energy(simulation)
        W_midpoint.append((E_after_midpoint - E_before_midpoint).value_in_unit(W_unit))

        # perform "reverse" protocol
        W_shads_R.append(measure_shadow_work(simulation, M))

    # use only the endpoints of each trajectory for DeltaF_neq estimation
    DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(np.array(W_shads_F)[:,-1], np.array(W_shads_R)[:,-1])

    return W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoint

def collect_and_save_results(schemes, simulation_factory, equilibrium_samples,
                             n_samples, M, midpoint_operator, temperature, name=""):
    """For each integrator scheme, perform n_samples nonequilibrium free energy measurements,
    and estimate the nonequilibrium free energy difference.

    * Saves results in {name}_results.pkl
    * Prints summary to console

    Returns
    """
    results = dict()
    for scheme in schemes:
        simulation = simulation_factory(scheme)
        results[scheme] = benchmark(simulation, equilibrium_samples, n_samples,
                           M=M, midpoint_operator=midpoint_operator, temperature=temperature)
        print("\n".join((scheme, get_summary_string(results[scheme], linebreaks=True))))

    with open("{}_results.pkl".format(name), "w") as f: dump(results, f)

    return results

if __name__ == '__main__':
    """Perform comparison of four Strang splittings of the Langevin equations, on
    a small system that can be quickly sampled on the Reference platform, and
    plot and save results.
    """
    platform = configure_platform("Reference")
    temperature = 298.0 * unit.kelvin
    burn_in_length = 100000 # in steps
    n_samples = 2000 # number of samples to collect
    protocol_length = 50 # length of protocol
    ghmc_thinning = protocol_length # number of GHMC steps between samples
    collision_rate = 91 / unit.picoseconds
    schemes = ["R V O V R", "O V R V O", "V R O R V", "V R R R O R R R V"]

    def test_on_alanine(schemes, constrained=True, randomize=True):
        """Compare the schemes on Alanine test system..."""

        # define test name
        if constrained: name = "constrained"
        else: name = "unconstrained"

        if randomize: name = name + "_randomized"
        else: name = name + "_null"
        print("\n\nName: {}".format(name))

        # get system
        if constrained:
            print("Constrained")
            testsystem = AlanineDipeptideVacuum(constraints=app.HBonds)
        else:
            print("Unconstrained")
            testsystem = AlanineDipeptideVacuum(constraints=None)

        (system, positions) = testsystem.system, testsystem.positions
        print("# atoms: {}".format(len(positions)))

        # if constrained, use a larger timestep
        if constrained: timestep = 2.5 * unit.femtoseconds
        else: timestep = 2.0 * unit.femtoseconds

        print("Timestep: {}".format(timestep))

        if randomize:
            midpoint_operator = lambda simulation:\
                randomization_midpoint_operator(simulation, temperature)
            print('Midpoint operator: randomization')

        if not randomize:
            midpoint_operator = null_midpoint_operator
            print('Midpoint operator: null')

        # define unbiased simulation...
        ghmc_timestep = 1.5 * unit.femtoseconds
        ghmc = GHMCIntegrator(temperature, timestep=ghmc_timestep)
        get_acceptance_rate = lambda: ghmc.getGlobalVariableByName("naccept")\
                          / ghmc.getGlobalVariableByName("ntrials")

        unbiased_simulation = app.Simulation(testsystem.topology, system, ghmc, platform)

        unbiased_simulation.context.setPositions(positions)
        unbiased_simulation.context.setVelocitiesToTemperature(temperature)
        print('"Burning in" unbiased GHMC sampler for {:.3}ps...'.format(
                (burn_in_length * ghmc_timestep).value_in_unit(unit.picoseconds)))
        unbiased_simulation.step(burn_in_length)
        print("Burn-in GHMC acceptance rate: {:.3f}%".format(100 * get_acceptance_rate()))
        ghmc.setGlobalVariableByName("naccept", 0)
        ghmc.setGlobalVariableByName("ntrials", 0)

        print("Collecting equilibrium samples...")
        equilibrium_samples = []
        for _ in range(5 * n_samples):
            unbiased_simulation.step(ghmc_thinning)
            equilibrium_samples.append(
                unbiased_simulation.context.getState(getPositions=True).getPositions(asNumpy=True))
        print("Equilibrated GHMC acceptance rate: {:.3f}%".format(100 * get_acceptance_rate()))

        def simulation_factory(scheme):
            """Factory for biased simulations"""
            lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                              collision_rate=collision_rate)

            simulation = app.Simulation(testsystem.topology, system, lsi, platform)
            simulation.context.setPositions(positions)
            simulation.context.setVelocitiesToTemperature(temperature)
            return simulation

        results = collect_and_save_results(schemes, simulation_factory, equilibrium_samples, n_samples,
                                 M=protocol_length, midpoint_operator=midpoint_operator, temperature=temperature,
                                 name=name)
        plot(results, name)

    for randomize in [True, False]:
        for constrained in [True, False]:
            test_on_alanine(schemes, constrained, randomize)