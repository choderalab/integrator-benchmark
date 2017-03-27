
from benchmark.testsystems import EquilibriumSimulator, NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.testsystems.coupled_power_oscillators import CoupledPowerOscillators
from benchmark.testsystems.configuration import configure_platform
from tqdm import tqdm

temperature = 298 * unit.kelvin

# Varying: well depth, well steepness, and same for the coupling terms

def oscillator_factory(coupling_strength=0.0):

    testsystem = CoupledPowerOscillators(nx=10, ny=10, nz=10, K=10000, b=12.0, well_radius=0.5,# bond_well_radius=1.0 / (1 + coupling_strength),
                                         coupling_strength=coupling_strength * unit.kilocalories_per_mole / unit.angstrom)
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    coupled_power_oscillators = EquilibriumSimulator(platform=configure_platform("CPU"),
                                               topology=top, system=sys, positions=pos,
                                               temperature=temperature,
                                               ghmc_timestep=0.5 * unit.femtosecond,
                                               burn_in_length=1000, n_samples=1000,
                                               thinning_interval=10, name="coupled_power_oscillators_{}".format(coupling_strength))
    return coupled_power_oscillators

if __name__ == "__main__":
    n_protocol_samples, protocol_length = 100, 100
    target_filename = os.path.join(DATA_PATH, "self_averaging_experiment.pkl")
    scheme = "V R O R V"


    results = {}
    coupling_strengths = np.linspace(0.0, 10000.0, 10)
    for coupling_strength in coupling_strengths:
        eq_sim = oscillator_factory(coupling_strength)
        integrator = LangevinSplittingIntegrator(splitting=scheme, temperature=temperature, timestep=3.5 * unit.femtosecond)
        noneq_sim = NonequilibriumSimulator(eq_sim, integrator)

        W_shads = np.zeros(n_protocol_samples)
        for i in tqdm(range(n_protocol_samples)):
            x_0 = eq_sim.sample_x_from_equilibrium()
            v_0 = eq_sim.sample_v_from_equilibrium()
            W_shads[i] = noneq_sim.accumulate_shadow_work(x_0, v_0, protocol_length)

        print("\n\nmean(W) = {:.3f}, stdev(W) = {:.3f}\n\n".format(np.mean(W_shads), np.std(W_shads)))
        results[coupling_strength] = W_shads


    with open(target_filename, "w") as f:
        pickle.dump(results, f)
