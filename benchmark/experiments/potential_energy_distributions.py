# in this simple test, we compare the sampled potential energy distribution
# with the equilibrium potential energy distribution for a variety of schemes

import matplotlib

from code.testsystems import system_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.experiments.benchmark import get_equilibrium_samples
from benchmark.integrators.integrators import LangevinSplittingIntegrator
from simtk.openmm import app
from simtk import unit
import numpy as np
from tqdm import tqdm
from code.utils import stderr
histstyle = {"alpha":0.3,
             "histtype":"stepfilled",
             "bins":50
             }

def plot_hist(array, label=""):
    plt.hist(array, label=label, **histstyle)

if __name__ == "__main__":
    schemes = ["V R O R V", "O R V R O",
               "R V O V R", "O V R V O",
               # "R R V O V R R", "O V R R R R V O",
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
    #n_samples = 100000
    thinning_interval = M

    _, unbiased_simulation = get_equilibrium_samples(topology, system, positions, platform,
                                                                       temperature=temperature,
                                                                       n_samples=100,
                                                                       thinning_interval=1,
                                                                       burn_in_length=n_samples * thinning_interval)
    positions = unbiased_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

    def simulation_factory(timestep, scheme):
        """Factory for biased simulations"""
        lsi = LangevinSplittingIntegrator(scheme, timestep=timestep, temperature=temperature,
                                          collision_rate=collision_rate)

        simulation = app.Simulation(topology, system, lsi, platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        return simulation

    def get_samples(simulation):
        samples = []
        energies = []
        for _ in tqdm(range(1000000)):
            simulation.step(1)
            state = simulation.context.getState(#getPositions=True,
                                                getEnergy=True)
            #x = state.getPositions(asNumpy=True)
            e = state.getPotentialEnergy()
            #samples.append(x.value_in_unit(x.unit))
            energies.append(e.value_in_unit(e.unit))
        return samples, energies

    results = {}
    samples, energies = get_samples(unbiased_simulation)
    results["GHMC"] = samples, energies
    ref = np.mean(energies)
    print("Average energy: {:.3f} +/- {:.3f}".format(np.mean(energies), 1.96 * stderr(energies)))

    for scheme in schemes:
        print(scheme)
        simulation = simulation_factory(3.0 * unit.femtoseconds, scheme)
        samples, energies = get_samples(simulation)
        results[scheme] = samples, energies
        print("Average energy: {:.3f} +/- {:.3f}".format(np.mean(energies), 1.96 * stderr(energies)))
        print("Average error: {:.3f} +/- {:.3f}".format(np.mean(energies - ref), 1.96 * stderr(energies - ref)))


    for scheme in schemes:
        plt.figure()
        _, e = results[scheme]
        plot_hist(results["GHMC"][1], label="GHMC")
        plot_hist(e, label=scheme)

        plt.legend(loc="best", fancybox=True)
        plt.savefig("constrained_potential_energy_distribution_{}.jpg".format(scheme), dpi=300)
        plt.close()

    from pickle import dump

    with open("constrained_energies.pkl", "w") as f: dump(results, f)