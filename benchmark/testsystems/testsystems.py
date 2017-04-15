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


def get_src_implicit_test_system(temperature):
    testsystem = SrcImplicit()
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    platform = mm.Platform.getPlatformByName("OpenCL")
    platform.setPropertyDefaultValue('OpenCLPrecision', 'double')

    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=0.5 * unit.femtoseconds,
                                                           burn_in_length=500, n_samples=500, thinning_interval=5)
    test_system = TestSystem(samples, temperature, top, sys, platform)
    return test_system


def get_src_explicit_test_system(temperature):
    testsystem = SrcExplicit()
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    platform = mm.Platform.getPlatformByName("OpenCL")
    platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')

    samples, unbiased_simulation = get_equilibrium_samples(top, sys, pos, platform, temperature,
                                                           ghmc_timestep=0.5 * unit.femtoseconds,
                                                           burn_in_length=100, n_samples=100, thinning_interval=5)
    test_system = TestSystem(samples, temperature, top, sys, platform)
    return test_system

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

harmonic_oscillator_params = simple_params.copy()
harmonic_oscillator_params["loader"] = load_harmonic_oscillator

quartic_params = simple_params.copy()
quartic_params["loader"] = load_quartic_potential

mts_params = simple_params.copy()
mts_params["loader"] = load_mts_test

system_params = {
    "harmonic_oscillator": harmonic_oscillator_params,
    "quartic_potential": quartic_params,
    "mts_test": mts_params,
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
