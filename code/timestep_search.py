from simtk import unit
from simtk.openmm import app
import numpy as np
from benchmark import get_equilibrium_samples, \
    randomization_midpoint_operator, null_midpoint_operator, apply_protocol
from testsystems import system_params
from integrators import LangevinSplittingIntegrator
from analysis import estimate_nonequilibrium_free_energy
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_appropriate_timestep(simulation_factory,
                              equilibrium_samples,
                              M,
                              midpoint_operator,
                              temperature,
                              timestep_range,
                              DeltaF_neq_threshold=1.0,
                              max_samples=10000,
                              verbose=True
                              ):
    """Perform binary search* over the timestep range, trying to find
    the maximum timestep that results in DeltaF_neq that doesn't exceed threshold
    or have gross instability problems.

    (*Not-quite-binary-search: instead of deterministic comparisons,
    it performs hypothesis tests at regular intervals.)

    Sketch
    ------
    * Maintain an interval (min_timestep, max_timestep)
    * At each iteration:
        * timestep <- (min_timestep + max_timestep) / 2
        * Only simulate long enough to be confident that DeltaF_neq(timestep) != threshold.
            * If we're confident DeltaF_neq(timestep) > threshold, reduce max_timestep to current timestep.
            * If we're confident DeltaF_neq(timestep) < threshold, increase min_timestep to current timestep

    Parameters
    ----------
    simulation_factory: function
        accepts a timestep argument and returns a simulation equipped with an integrator with that
        timestep
    equilibrium_samples: list
        list of samples from the configuration distribution at equilibrium
    M: int
        protocol length
    midpoint_operator: function
        accepts a simulation as an argument, doesn't return anything
    temperature: unit'd quantity
        temperature used to resample velocities
    timestep_range: iterable
        (min_timestep, max_timestep)
    DeltaF_neq_threshold: double, default=1.0
        maximum allowable DeltaF_neq
    max_samples: int
        number of samples
    verbose: boolean
        if True, print a bunch of stuff to the command prompt

    Returns
    -------
    timestep: unit'd quantity
        Maximum timestep tested that doesn't exceed the DeltaF_neq_threshold
    """
    max_iter = 10
    batch_size = 100
    alpha = 1.96 # for now hard-coded confidence level

    min_timestep, max_timestep = timestep_range[0], timestep_range[-1]

    for i in range(max_iter):
        timestep = (min_timestep + max_timestep) / 2
        if verbose:
            print("Current feasible range: [{:.3f}fs, {:.3f}fs]".format(
                min_timestep.value_in_unit(unit.femtosecond),
                max_timestep.value_in_unit(unit.femtosecond)
            ))
            print("Testing: {:.3f}fs".format(timestep.value_in_unit(unit.femtosecond)))
        simulation = simulation_factory(timestep)
        simulation_crashed = False
        changed_timestep_range = False
        W_shads_F, W_shads_R, W_midpoints = [], [], []

        def update_lists(W_shad_F, W_midpoint, W_shad_R):
            W_shads_F.append(W_shad_F)
            W_midpoints.append(W_midpoint)
            W_shads_R.append(W_shad_R)

        # collect up to max_samples protocol samples, making a decision about whether to proceed
        # every batch_size samples
        for _ in range(max_samples / batch_size):

            # collect another batch_size protocol samples
            for _ in range(batch_size):
                # draw equilibrium sample
                #x, v = equilibrium_sampler()
                #simulation.context.setPositions(x)
                #simulation.context.setVelocities(v)
                simulation.context.setPositions(equilibrium_samples[np.random.randint(len(equilibrium_samples))])
                simulation.context.setVelocitiesToTemperature(temperature)

                # collect and store measurements
                # if the simulation crashes, set simulation_crashed flag
                try:
                    update_lists(*apply_protocol(simulation, M, midpoint_operator))
                except:
                    simulation_crashed = True
                    if verbose: print("A simulation crashed! Considering this timestep unstable...")

            # if we didn't crash, update estimate of DeltaF_neq upper and lower confidence bounds
            DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(np.array(W_shads_F)[:,-1], np.array(W_shads_R)[:,-1])
            if np.isnan(DeltaF_neq + sq_uncertainty):
                if verbose:
                    print("A simulation encountered NaNs!")
                simulation_crashed = True
            bound = alpha * np.sqrt(sq_uncertainty)
            DeltaF_neq_lcb, DeltaF_neq_ucb = DeltaF_neq - bound, DeltaF_neq + bound
            out_of_bounds = (DeltaF_neq_lcb > DeltaF_neq_threshold) or (DeltaF_neq_ucb < DeltaF_neq_threshold)

            if verbose and (out_of_bounds or simulation_crashed):
                print("After collecting {} protocol samples, DeltaF_neq is likely in the following interval: "
                "[{:.3f}, {:.3f}]".format(len(W_shads_F), DeltaF_neq_lcb, DeltaF_neq_ucb))

            # if (DeltaF_neq_lcb > threshold) or (nans are encountered), then we're pretty sure this timestep is too big,
            # and we can move on to try a smaller one
            if simulation_crashed or (DeltaF_neq_lcb > DeltaF_neq_threshold):
                if verbose:
                    print("This timestep is probably too big!\n")
                max_timestep = timestep
                changed_timestep_range = True
                break

            # else, if (DeltaF_neq_ucb < threshold), then we're pretty sure we can get
            # away with a larger timestep
            elif (DeltaF_neq_ucb < DeltaF_neq_threshold):
                if verbose:
                    print("We can probably get away with a larger timestep!\n")
                min_timestep = timestep
                changed_timestep_range = True
                break

            # else, the threshold is within the upper and lower confidence bounds, and we keep going

        if (not changed_timestep_range):
            timestep = (min_timestep + max_timestep) / 2
            if verbose:
                print("\nTerminating early: found the following timestep: ".format(timestep.value_in_unit(unit.femtosecond)))
            return timestep
    if verbose:
        timestep = (min_timestep + max_timestep) / 2
        print("\nTerminating: found the following timestep: ".format(timestep.value_in_unit(unit.femtosecond)))
    return timestep

def sweep_over_timesteps(simulation_factory,
                          equilibrium_samples,
                          M,
                          midpoint_operator,
                          timesteps_to_try,
                          temperature,
                          n_samples=10000,
                          verbose=True
                          ):

    DeltaF_neqs, sq_uncertainties = [], []

    for timestep in timesteps_to_try:
        simulation = simulation_factory(timestep)
        W_shads_F, W_shads_R, W_midpoints = [], [], []

        def update_lists(W_shad_F, W_midpoint, W_shad_R):
            W_shads_F.append(W_shad_F)
            W_midpoints.append(W_midpoint)
            W_shads_R.append(W_shad_R)

        # collect up to max_samples protocol samples, making a decision about whether to proceed
        # every batch_size samples
        for _ in tqdm(range(n_samples)):

            # draw equilibrium sample
            simulation.context.setPositions(equilibrium_samples[np.random.randint(len(equilibrium_samples))])
            simulation.context.setVelocitiesToTemperature(temperature)

            # collect and store measurements
            # if the simulation crashes, set simulation_crashed flag
            try:
                update_lists(*apply_protocol(simulation, M, midpoint_operator))
            except:
                simulation_crashed = True
                if verbose:
                    print("A simulation crashed! Considering this timestep unstable...")

        # if we didn't crash, update estimate of DeltaF_neq upper and lower confidence bounds
        DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(np.array(W_shads_F)[:,-1], np.array(W_shads_R)[:,-1])
        DeltaF_neqs.append(DeltaF_neq)
        sq_uncertainties.append(sq_uncertainty)
        print("\tTimestep: {:.3f}\n\tDeltaF_neq: {:.3f} +/- {:.3f}".format(
            timestep.value_in_unit(unit.femtosecond),
            DeltaF_neq,
            1.96 * np.sqrt(sq_uncertainty)
        ))
    return DeltaF_neqs, sq_uncertainties

if __name__ == "__main__":
    schemes = ["V R O R V", "O R V R O",
               "R V O V R", "O V R V O",
               "R R V O V R R", "O V R R R R V O",
               "V R R O R R V", "V R R R O R R R V"
               ]
    print(schemes)

    params = system_params["alanine"]
    topology, system, positions = params["loader"](constrained=True)
    temperature = params["temperature"]
    collision_rate = params["collision_rate"]
    platform = params["platform"]
    temperature = params["temperature"]
    M = params["protocol_length"]
    n_samples = params["n_samples"]
    thinning_interval = M

    midpoint_operator = lambda simulation: randomization_midpoint_operator(simulation, temperature)
    #midpoint_operator = null_midpoint_operator

    equilibrium_samples, unbiased_simulation = get_equilibrium_samples(topology, system, positions, platform,
                                                                       temperature=temperature,
                                                                       n_samples=n_samples,
                                                                       thinning_interval=thinning_interval,
                                                                       burn_in_length=n_samples * thinning_interval)

    def simulation_factory(timestep, scheme):
        """Factory for biased simulations"""
        lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                          collision_rate=collision_rate)

        simulation = app.Simulation(topology, system, lsi, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        return simulation

    # plot DeltaF_neq as function of timestep
    timesteps_to_try = np.linspace(0.25,5,10) * unit.femtosecond
    results = {}
    plt.figure()
    for scheme in schemes:
        print(scheme)
        sim_factory = lambda timestep: simulation_factory(timestep, scheme)
        DeltaFs, sq_uncertainties = sweep_over_timesteps(sim_factory, equilibrium_samples, M,
                             midpoint_operator, timesteps_to_try, temperature,  n_samples=n_samples)
        results[scheme] = DeltaFs, sq_uncertainties
        sigmas = 1.96 * np.sqrt(sq_uncertainties)
        # plot the results
        plt.errorbar(timesteps_to_try.value_in_unit(unit.femtosecond), DeltaFs, yerr=sigmas, label=scheme)

    plt.xlabel("Timestep (fs)")
    plt.ylabel("$\Delta F_{neq}$")
    plt.legend(loc='best', fancybox=True)
    plt.savefig("../figures/alanine_conf_DeltaFs.jpg")
    plt.close()

    # search for max allowable timestep
    for scheme in schemes:
        print("\n" + scheme)
        sim_factory = lambda timestep: simulation_factory(timestep, scheme)
        timestep = find_appropriate_timestep(sim_factory, equilibrium_samples,
                                  M=M, midpoint_operator=midpoint_operator, temperature=temperature,
                                  timestep_range=[0.1 * unit.femtosecond, 10 * unit.femtosecond], max_samples=n_samples
                                  )
        print(timestep)
