import numpy as np
from openmmtools.testsystems import SrcImplicit, DHFRExplicit
from simtk.openmm import app
from simtk import unit
from benchmark.testsystems.configuration import configure_platform
from benchmark.utilities import keep_only_some_forces


def load_src_vacuum(constrained=True):
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = SrcImplicit(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system)

    return topology, system, positions

def load_src_implicit(constrained=True):
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = SrcImplicit(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system, extra_forces_to_keep=["GBSAOBCForce"])

    return topology, system, positions

def load_dhfr_explicit(constrained=True):
    if constrained:
        constraints = app.HBonds
        rigid_water = True
    else:
        constraints = None
        rigid_water = False


    testsystem = DHFRExplicit(constraints=constraints, rigid_water=rigid_water)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system)

    return topology, system, positions

simple_params = {
    "platform": configure_platform("Reference"),
    "burn_in_length": 1000,
    "n_samples": 10000,
    "protocol_length": 50,
    "constrained_timestep": 2.5*unit.femtosecond,
    "unconstrained_timestep": 2.0*unit.femtosecond,
    "temperature": 298.0 * unit.kelvin,
    "collision_rate": 91 / unit.picoseconds,
}

#harmonic_oscillator_params = simple_params.copy()
#from .low_dimensional_systems import load_harmonic_oscillator
#harmonic_oscillator_params["loader"] = load_harmonic_oscillator

quartic_params = simple_params.copy()
from .low_dimensional_systems import load_quartic_potential
quartic_params["loader"] = load_quartic_potential

#mts_params = simple_params.copy()
#from .low_dimensional_systems import load_mts_test
#mts_params["loader"] = load_mts_test

from .waterbox import load_waterbox
from .alanine_dipeptide import load_alanine
system_params = {
    #"harmonic_oscillator": harmonic_oscillator_params,
    "quartic_potential": quartic_params,
    #"mts_test": mts_params,
    "waterbox": {
        "platform" : configure_platform("OpenCL"),
        "loader": load_waterbox,
        "burn_in_length": 100,
        "n_samples": 50,
        "protocol_length": 50,
        "constrained_timestep": 2.5*unit.femtosecond,
        "unconstrained_timestep": 1.0*unit.femtosecond,
        "temperature": 298.0 * unit.kelvin,
        "collision_rate": 91 / unit.picoseconds,
    },
    "alanine": {
        "platform": configure_platform("Reference"),
        "loader": load_alanine,
        "burn_in_length": 1000,
        "n_samples": 1000,
        "protocol_length": 100,
        "constrained_timestep": 2.5*unit.femtosecond,
        "unconstrained_timestep": 2.0*unit.femtosecond,
        "temperature": 298.0 * unit.kelvin,
        "collision_rate": 91 / unit.picoseconds,
    }
}
# TODO: Add Waterbox, AlanineExplicit EquilibriumSimulators
