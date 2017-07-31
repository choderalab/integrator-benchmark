from simtk import unit

from benchmark import simulation_parameters
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems.alanine_dipeptide import load_alanine
from benchmark.testsystems.bookkeepers import NonequilibriumSimulator
from benchmark.testsystems.configuration import configure_platform
import numpy as np

n_samples = 100
thinning_interval = 100

temperature = simulation_parameters["temperature"]
from benchmark.testsystems.bookkeepers import EquilibriumSimulator

top, sys, pos = load_alanine(constrained=True)
alanine_constrained = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=500, n_samples=n_samples,
                                           thinning_interval=thinning_interval, name="alanine_constrained_test")
print("Sample from cache: (xyz, periodic_box_vectors)\n", alanine_constrained.sample_x_from_equilibrium())

sim = NonequilibriumSimulator(alanine_constrained,
                              LangevinSplittingIntegrator("O V R V O", timestep=4.5 * unit.femtoseconds))
result = sim.collect_protocol_samples(100, 100, store_potential_energy_traces=True, store_W_shad_traces=True)
print("<W>_(0 -> M): {:.3f}, <W>_(M -> 2M): {:.3f}".format(result["W_shads_F"].mean(), result["W_shads_R"].mean()))

forward_traces = np.array([t[0] for t in result["W_shad_traces"]])
print("0 -> M traces: mean W_shad per step: ", forward_traces.mean(0))
print("0 -> M traces: mean cumulative W_shad per step: ", np.cumsum(forward_traces,1).mean(0))
