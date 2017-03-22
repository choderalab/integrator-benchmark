# alanine dipeptide in vacuum, implicit solvent, and explicit solvent

import numpy as np
from openmmtools.testsystems import CustomExternalForcesTestSystem, AlanineDipeptideVacuum, WaterBox, AlanineDipeptideExplicit, SrcImplicit
from simtk.openmm import app
from simtk import unit
from configuration import configure_platform
from benchmark.utilities import keep_only_some_forces


def load_alanine(constrained=True):
    """Load AlanineDipeptide vacuum, optionally with hydrogen bonds constrained"""
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = AlanineDipeptideVacuum(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system)

    return topology, system, positions

def load_solvated_alanine(constrained=True):
    """Load AlanineDipeptide in explicit solvent,
    optionally with rigid water + hydrogen bonds constrained."""
    args = {"ewaldErrorTolerance":1e-5,
            "nonbondedCutoff":10*unit.angstroms
            }
    if constrained:
        testsystem = AlanineDipeptideExplicit(constraints=app.HBonds, rigid_water=True, **args)
    else:
        testsystem = AlanineDipeptideExplicit(constraints=None, rigid_water=False, **args)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions
    return topology, system, positions


temperature = 298 * unit.kelvin
from bookkeepers import EquilibriumSimulator
top, sys, pos = load_alanine(constrained=True)
alanine_constrained = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=2.5 * unit.femtosecond,
                                           burn_in_length=1000, n_samples=10000,
                                           thinning_interval=100, name="alanine_constrained")

top, sys, pos = load_alanine(constrained=False)
alanine_unconstrained = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=2.0 * unit.femtosecond,
                                           burn_in_length=1000, n_samples=10000,
                                           thinning_interval=100, name="alanine_unconstrained")