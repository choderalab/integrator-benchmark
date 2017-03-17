import numpy as np
from openmmtools.testsystems import CustomExternalForcesTestSystem, AlanineDipeptideVacuum, WaterBox, AlanineDipeptideExplicit, SrcImplicit
from simtk.openmm import app
from simtk import unit
from configuration import configure_platform
from benchmark.utilities import keep_only_some_forces
from benchmark import DATA_PATH

def load_waterbox(constrained=True):
    """Load WaterBox test system with non-default PME cutoff and error tolerance"""
    testsystem = WaterBox(constrained=constrained, ewaldErrorTolerance=1e-5, cutoff=10*unit.angstroms)
    (topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions
    return topology, system, positions


temperature = 298 * unit.kelvin
from bookkeepers import EquilibriumSimulator
top, sys, pos = load_waterbox(constrained=True)
waterbox_constrained = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=2.0 * unit.femtosecond,
                                           burn_in_length=100, n_samples=100,
                                           thinning_interval=10, name="waterbox_constrained")

top, sys, pos = load_waterbox(constrained=False)
flexible_waterbox = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100, n_samples=100,
                                           thinning_interval=10, name="flexible_waterbox")
