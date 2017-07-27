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
                                           ghmc_timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="waterbox_constrained")

top, sys, pos = load_waterbox(constrained=False)
flexible_waterbox = EquilibriumSimulator(platform=configure_platform("CUDA"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=0.5 * unit.femtosecond,
                                           burn_in_length=200000, n_samples=1000,
                                           thinning_interval=20000, name="flexible_waterbox")
