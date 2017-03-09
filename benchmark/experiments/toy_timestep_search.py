import matplotlib

from benchmark.testsystems.testsystems import system_params
from timestep_search import sweep_over_timesteps

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.experiments.benchmark import randomization_midpoint_operator, get_equilibrium_samples
from benchmark.integrators.integrators import LangevinSplittingIntegrator
from simtk.openmm import app
from simtk import unit
import numpy as np
from code.utils import generate_solvent_solute_splitting_string

if __name__=="__main__":
    schemes = [generate_solvent_solute_splitting_string(K_p=2, K_r=1),
               "V R O R V", "O R V R O",
               "R V O V R", "O V R V O"
               # "R R V O V R R", "O V R R R R V O",
               # "V R R O R R V", "V R R R O R R R V"
               ]
    print(schemes)

    params = system_params["mts_test"]
    topology, system, positions = params["loader"](constrained=True)
    temperature = params["temperature"]
    collision_rate = params["collision_rate"]
    platform = params["platform"]
    temperature = params["temperature"]
    M = params["protocol_length"]
    n_samples = params["n_samples"] / 10
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
    def plot_deltaF_neq(schemes, timesteps_to_try):
        results = {}
        plt.figure()
        for scheme in schemes:
            print(scheme)
            sim_factory = lambda timestep: simulation_factory(timestep, scheme)
            DeltaFs, sq_uncertainties = sweep_over_timesteps(sim_factory, equilibrium_samples, M,
                                                             midpoint_operator, timesteps_to_try, temperature,
                                                             n_samples=n_samples)
            results[scheme] = DeltaFs, sq_uncertainties
            sigmas = 1.96 * np.sqrt(sq_uncertainties)
            # plot the results
            plt.errorbar(timesteps_to_try.value_in_unit(unit.femtosecond), DeltaFs, yerr=sigmas, label=scheme)

        plt.xlabel("Timestep (fs)")
        plt.ylabel("$\Delta F_{neq}$")
        plt.legend(loc='best', fancybox=True)

    plot_deltaF_neq(schemes, np.linspace(50,100,10) * unit.femtosecond)
    plt.savefig("../figures/mts_test_conf_DeltaFs.jpg")
    plt.close()