from simtk import unit
from simtk.openmm import app
import numpy as np
from benchmark import system_params, get_equilibrium_samples, randomization_midpoint_operator, apply_protocol
from integrators import LangevinSplittingIntegrator
from analysis import estimate_nonequilibrium_free_energy

def find_appropriate_timestep(simulation_factory,
                              equilibrium_sampler,
                              M,
                              midpoint_operator,
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
    equilibrium_sampler: function
        no arguments, returns a (x,v)
       pair that's drawn from the correct equilibrium distribution
    M: int
        protocol length
    midpoint_operator: function
        accepts a simulation as an argument, doesn't return anything
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
                x, v = equilibrium_sampler()
                simulation.context.setPositions(x)
                simulation.context.setVelocities(v)

                # collect and store measurements
                # if the simulation crashes, set simulation_crashed flag
                try:
                    update_lists(*apply_protocol(simulation, M, midpoint_operator))
                except:
                    simulation_crashed = True
                    if verbose: print("A simulation crashed! Considering this timestep unstable...")

            # if we didn't crash, update estimate of DeltaF_neq upper and lower confidence bounds
            DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(np.array(W_shads_F)[:,-1], np.array(W_shads_R)[:,-1])
            if np.isnan(DeltaF_neq + sq_uncertainty): simulation_crashed = True
            bound = alpha * np.sqrt(sq_uncertainty)
            DeltaF_neq_lcb, DeltaF_neq_ucb = DeltaF_neq - bound, DeltaF_neq + bound

            if verbose and ((DeltaF_neq_lcb > DeltaF_neq_threshold) or (DeltaF_neq_ucb < DeltaF_neq_threshold) or simulation_crashed):
                print("After collecting {} protocol samples, DeltaF_neq is likely in the following interval: "
                "[{:.3f}, {:.3f}]".format(len(W_shads_F), DeltaF_neq_lcb, DeltaF_neq_ucb))

            # if (DeltaF_neq_lcb > threshold) or (nans are encountered), then we're pretty sure this timestep is too big,
            # and we can move on to try a smaller one
            if simulation_crashed or (DeltaF_neq_lcb > DeltaF_neq_threshold):
                if verbose: print("This timestep is probably too big!\n")
                max_timestep = timestep
                break

            # else, if (DeltaF_neq_ucb < threshold), then we're pretty sure we can get
            # away with a larger timestep
            elif (DeltaF_neq_ucb < DeltaF_neq_threshold):
                if verbose: print("We can probably get away with a larger timestep!\n")
                min_timestep = timestep
                break

            # else, the threshold is within the upper and lower confidence bounds, and we keep going
    if verbose: print("\nTerminating: found the following timestep: ".format(timestep.value_in_unit(unit.femtosecond)))
    return timestep

if __name__ == "__main__":
    # scheme = "V R R R O R R R V"
    scheme = "O V R V O"
    print(scheme)

    params = system_params["alanine"]
    topology, system, positions = params["loader"](constrained=True)
    temperature = params["temperature"]
    collision_rate = params["collision_rate"]
    platform = params["platform"]
    temperature = params["temperature"]
    M = params["protocol_length"]
    n_samples = 10000
    thinning_interval = M

    midpoint_operator = lambda simulation: randomization_midpoint_operator(simulation, temperature)

    equilibrium_samples, unbiased_simulation = get_equilibrium_samples(topology, system, positions, platform,
                                                                       temperature=temperature,
                                                                       n_samples=n_samples,
                                                                       thinning_interval=thinning_interval,
                                                                       burn_in_length=n_samples * thinning_interval)


    def equilibrium_sampler():
        x = equilibrium_samples[np.random.randint(len(equilibrium_samples))]
        unbiased_simulation.context.setVelocitiesToTemperature(temperature)
        v = unbiased_simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
        return x, v


    def simulation_factory(timestep):
        """Factory for biased simulations"""
        lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                          collision_rate=collision_rate)

        simulation = app.Simulation(topology, system, lsi, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        return simulation

    find_appropriate_timestep(simulation_factory, equilibrium_sampler,
                              M=params["protocol_length"], midpoint_operator=midpoint_operator,
                              timestep_range=[0.1 * unit.femtosecond, 8 * unit.femtosecond], max_samples=n_samples
                              )
