from openmmtools.testsystems import WaterBox
from simtk import unit
from benchmark.testsystems.configuration import configure_platform
from benchmark import simulation_parameters
from benchmark.utilities import add_barostat


def load_waterbox(constrained=True):
    """Load WaterBox test system"""
    testsystem = WaterBox(constrained=constrained)
    (topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions
    add_barostat(system)
    return topology, system, positions


temperature = simulation_parameters["temperature"]
from benchmark.testsystems.bookkeepers import EquilibriumSimulator
top, sys, pos = load_waterbox(constrained=True)
waterbox_constrained = EquilibriumSimulator(platform=configure_platform("CUDA"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="waterbox_constrained")

top, sys, pos = load_waterbox(constrained=False)
flexible_waterbox = EquilibriumSimulator(platform=configure_platform("CUDA"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           timestep=0.5 * unit.femtosecond,
                                           burn_in_length=200000, n_samples=1000,
                                           thinning_interval=20000, name="flexible_waterbox")

# add a smaller waterbox, for the purposes of comparison of the inefficient nested estimator
# against the efficient near-equilibrium estimator
small_box_edge = 15 * unit.angstrom
testsystem = WaterBox(box_edge=small_box_edge,
                      cutoff=small_box_edge / 2.5, # box_edge must be > (2 * cutoff)
                      )
(top, sys, pos) = testsystem.topology, testsystem.system, testsystem.positions
add_barostat(sys)

tiny_waterbox = EquilibriumSimulator(platform=configure_platform("CUDA"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="tiny_waterbox_constrained")
