import numpy as np
from simtk import unit
from simtk.openmm import app

from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import system_params

W_unit = unit.kilojoule_per_mole
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from pickle import dump
from openmmtools.integrators import GHMCIntegrator
from tqdm import tqdm

from benchmark.plotting import plot
from benchmark.utilities import get_total_energy, get_summary_string, measure_shadow_work


def randomization_midpoint_operator(simulation, temperature):
    """Resamples velocities from Maxwell-Boltzmann distribution."""
    simulation.context.setVelocitiesToTemperature(temperature)

def null_midpoint_operator(simulation):
    """Do nothing to the simulation"""
    pass

def apply_protocol(simulation, M, midpoint_operator):
    # perform "forward" protocol
    W_shad_F = measure_shadow_work(simulation, M)

    # perform midpoint operation
    E_before_midpoint = get_total_energy(simulation)
    midpoint_operator(simulation)
    E_after_midpoint = get_total_energy(simulation)
    W_midpoint = (E_after_midpoint - E_before_midpoint).value_in_unit(W_unit)

    # perform "reverse" protocol
    W_shad_R = measure_shadow_work(simulation, M)

    return W_shad_F, W_midpoint, W_shad_R

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
    W_shads_F, W_midpoints, W_shads_R = [], [], []
    def update_lists(W_shad_F, W_midpoint, W_shad_R):
        W_shads_F.append(W_shad_F)
        W_midpoints.append(W_midpoint)
        W_shads_R.append(W_shad_R)

    for _ in tqdm(range(n_samples)):
        # draw equilibrium sample
        simulation.context.setPositions(equilibrium_samples[np.random.randint(len(equilibrium_samples))])
        simulation.context.setVelocitiesToTemperature(temperature)

        # collect and store measurements
        update_lists(*apply_protocol(simulation, M, midpoint_operator))

    # use only the endpoints of each trajectory for DeltaF_neq estimation
    DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(np.array(W_shads_F)[:,-1], np.array(W_shads_R)[:,-1])

    return W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoints

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

def build_benchmark(system_name):
    assert(system_name in system_params.keys())
    params = system_params[system_name]

    def tester(schemes, constrained=True, randomize=True):
        """Function that accepts a list of splitting scheme strings and

        Pickles results and outputs a bunch of figures

        Parameters
        ----------
        schemes : list of strings
            List of splitting scheme strings
        constrained : boolean

        randomize : boolean
            If true, randomize the velocities at the midpoint
            Else, do nothing


        """
        # define test name
        constraint_string = {True: "constrained", False: "unconstrained"}
        midpoint_string = {True: "randomized", False: "null"}
        name = "_".join([system_name, constraint_string[constrained], midpoint_string[randomize]])
        print("\n\nName: {}".format(name))

        # load the system
        topology, system, positions = params["loader"](constrained)

        # print the active forces
        for force in system.getForces(): print(type(force))

        # set the timestep
        if constrained: timestep = params["constrained_timestep"]
        else: timestep = params["unconstrained_timestep"]
        print("Timestep: {}".format(timestep))

        # set the midpoint operator
        if randomize:
            midpoint_operator = lambda simulation: \
                randomization_midpoint_operator(simulation, params["temperature"])
            print('Midpoint operator: randomization')

        if not randomize:
            midpoint_operator = null_midpoint_operator
            print('Midpoint operator: null')

        n_samples, thinning_interval = params["n_samples"], params["protocol_length"]
        equilibrium_samples, unbiased_sim = get_equilibrium_samples(topology, system, positions,
                                                      thinning_interval=thinning_interval,
                                                      **params)
        # define a factory for Langevin simulations
        def simulation_factory(scheme, timestep=timestep):
            """Factory for biased simulations"""
            lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=params["temperature"],
                                              collision_rate=params["collision_rate"])

            simulation = app.Simulation(topology, system, lsi, params["platform"])
            simulation.context.setPositions(positions)
            simulation.context.setVelocitiesToTemperature(params["temperature"])
            return simulation

        print("Collecting {} protocol samples per condition...".format(params["n_samples"]))
        results = collect_and_save_results(schemes, simulation_factory, equilibrium_samples, params["n_samples"],
                                           M=params["protocol_length"], midpoint_operator=midpoint_operator,
                                           temperature=params["temperature"],
                                           name=name)
        plot(results, name)

    return tester

if __name__ == '__main__':
    """Perform comparison of several Strang splittings of the Langevin equations, on
    small systems that can be quickly sampled, and plot and save results.
    """
    schemes = ["R V O V R", "O V R V O", "V R O R V", "V R R R O R R R V"]

    print("Testing on AlanineDipeptideVecuum")
    test = build_benchmark(system_name="alanine")
    for constrained in [True, False]:
        for randomize in [True, False]:
            test(schemes, constrained, randomize)

    print("Testing on WaterBox")
    test = build_benchmark(system_name="waterbox")
    for constrained in [True, False]:
        for randomize in [True, False]:
            test(schemes, constrained, randomize)
