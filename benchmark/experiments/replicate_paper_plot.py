import matplotlib
import numpy as np
from simtk import unit
from simtk.openmm import app

from benchmark.experiments.benchmark import get_equilibrium_samples, null_midpoint_operator
from benchmark.integrators.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import system_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.experiments.timestep_search import sweep_over_timesteps

if __name__ == "__main__":
    unconstrained_timesteps_to_try = [0.5, 0.75, 1, 1.25, 1.5]
    constrained_timesteps_to_try = [0.5, 2, 3, 4, 5, 6]
    unconstrained_timesteps_to_try = np.array(unconstrained_timesteps_to_try) * unit.femtosecond
    constrained_timesteps_to_try = np.array(constrained_timesteps_to_try) * unit.femtosecond


    def simulation_factory(timestep, scheme):
        """Factory for biased simulations"""
        lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                          collision_rate=collision_rate)

        simulation = app.Simulation(topology, system, lsi, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        return simulation


    # plot DeltaF_neq as function of timestep
    def plot_deltaF_neq(schemes, timesteps_to_try, midpoint_operator, constrained, n_waters):
        results = {}
        for scheme in schemes:
            print(scheme)
            sim_factory = lambda timestep: simulation_factory(timestep, scheme)
            DeltaFs, sq_uncertainties = sweep_over_timesteps(sim_factory, equilibrium_samples, M,
                                                             midpoint_operator, timesteps_to_try, temperature,
                                                             n_samples=n_samples)
            results[scheme] = DeltaFs, sq_uncertainties
            sigmas = 1.96 * np.sqrt(sq_uncertainties)
            # plot the results
            c_string = " (unconstrained)"
            if constrained:
                c_string = " (constrained)"
            plt.errorbar(timesteps_to_try.value_in_unit(unit.femtosecond), np.array(DeltaFs) / n_waters,
                         yerr=sigmas / n_waters, label=scheme + c_string)

        plt.xlabel("Timestep (fs)")
        plt.ylabel(r"$\Delta F_{neq} / (N_{H 2 O} k_B T)$")
        plt.legend(loc='best', fancybox=True)


    schemes = ["O V R V O"]

    params = system_params["waterbox"]
    temperature = params["temperature"]
    collision_rate = params["collision_rate"]
    platform = params["platform"]
    temperature = params["temperature"]

    # over-riding these, just for speed...
    # M = params["protocol_length"]
    M = 50
    # n_samples = params["n_samples"]
    n_samples = 50

    thinning_interval = M

    midpoint_operator = null_midpoint_operator
    plt.figure()
    for constrained in [True, False]:
        if constrained:
            ts = constrained_timesteps_to_try
            ghmc_timestep = 1.5 * unit.femtosecond
        else:
            ts = unconstrained_timesteps_to_try
            ghmc_timestep = 1.0 * unit.femtosecond

        topology, system, positions = params["loader"](constrained=constrained)

        equilibrium_samples, unbiased_simulation = get_equilibrium_samples(topology, system, positions, platform,
                                                                           temperature=temperature, burn_in_length=100,
                                                                           n_samples=n_samples,
                                                                           thinning_interval=thinning_interval,
                                                                           ghmc_timestep=ghmc_timestep
                                                                           )

        n_waters = len(positions) / 3.0
        plot_deltaF_neq(schemes, ts, midpoint_operator, constrained, n_waters)

    plt.savefig("../figures/waterbox_full_DeltaFs.jpg")
    plt.close()
