# quick test: how does the estimated DeltaF_neq for a fixed timestep
# stabilize as we vary the number of protocol steps M?



# fix test system and integrator


import matplotlib
import numpy as np
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm

from benchmark.experiments.benchmark import estimate_nonequilibrium_free_energy
from benchmark.experiments.benchmark import get_equilibrium_samples, \
    randomization_midpoint_operator, apply_protocol
from benchmark.integrators.integrators import LangevinSplittingIntegrator
from code.testsystems import system_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sweep_over_Ms(simulation_factory,
                          equilibrium_samples,
                          Ms,
                          midpoint_operator,
                          timestep,
                          temperature,
                          n_samples=10000,
                          verbose=True
                          ):

    DeltaF_neqs, sq_uncertainties = [], []

    for M in Ms:
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
        W_F = np.array(W_shads_F)[:,-1]
        W_R = np.array(W_shads_R)[:, -1]
        DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(W_F, W_R)
        DeltaF_neqs.append(DeltaF_neq)
        sq_uncertainties.append(sq_uncertainty)
        print("\tProtocol length: {}\n\tEstimated DeltaF_neq: {:.3f} +/- {:.3f}".format(
            M,
            DeltaF_neq,
            1.96 * np.sqrt(sq_uncertainty)
        ))
    return DeltaF_neqs, sq_uncertainties

if __name__ == "__main__":

    Ms = [1, 2, 3, 4, 5, 10, 20, 50]


    schemes = [#"V R O R V", # BAOAB (small)
               "O V R V O", # VVVR (large)
               ]
    print(schemes)
    constrained = False
    name = "waterbox"
    params = system_params[name]
    topology, system, positions = params["loader"](constrained=constrained)
    temperature = params["temperature"]
    collision_rate = params["collision_rate"]
    platform = params["platform"]
    temperature = params["temperature"]
    M = params["protocol_length"]
    n_samples = params["n_samples"]
    thinning_interval = M

    timestep = 2.5 * unit.femtosecond

    equilibrium_samples, unbiased_simulation = get_equilibrium_samples(topology, system, positions, platform,
                                                                       temperature=temperature,
                                                                       n_samples=n_samples*5,
                                                                       thinning_interval=thinning_interval,
                                                                       burn_in_length=n_samples * thinning_interval,
                                                                       ghmc_timestep=0.5*unit.femtosecond)

    def simulation_factory(timestep, scheme):
        """Factory for biased simulations"""
        lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                          collision_rate=collision_rate)

        simulation = app.Simulation(topology, system, lsi, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        return simulation

    # plot DeltaF_neq as function of timestep
    def plot_deltaF_neq(schemes, Ms, midpoint_operator):
        results = {}
        plt.figure()
        for scheme in schemes:
            print(scheme)
            sim_factory = lambda timestep: simulation_factory(timestep, scheme)
            DeltaFs, sq_uncertainties = sweep_over_Ms(sim_factory, equilibrium_samples, Ms,
                                 midpoint_operator, timestep, temperature,  n_samples=n_samples)
            results[scheme] = DeltaFs, sq_uncertainties
            sigmas = 1.96 * np.sqrt(sq_uncertainties)
            # plot the results
            plt.errorbar(Ms, DeltaFs, yerr=sigmas, label=scheme)

        plt.xlabel("Protocol length ($M$)")
        plt.ylabel("$\Delta F_{neq}$")
        plt.legend(loc='best', fancybox=True)

    plot_deltaF_neq(schemes, Ms,
                    lambda simulation: randomization_midpoint_operator(simulation, temperature))
    plt.savefig("../figures/{}_conf_DeltaFs_over_Ms.jpg".format(name))
    plt.close()