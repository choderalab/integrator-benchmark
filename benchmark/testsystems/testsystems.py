import numpy as np
from openmmtools.testsystems import LysozymeImplicit, DHFRExplicit
from simtk.openmm import app
from simtk import unit
from benchmark.testsystems.configuration import configure_platform
from benchmark.utilities import keep_only_some_forces
from benchmark import simulation_parameters
from benchmark.utilities import add_barostat
from .low_dimensional_systems import load_constraint_coupled_harmonic_oscillators
temperature = simulation_parameters["temperature"]

def load_t4_implicit(constrained=True):
    if constrained:
        constraints = app.HBonds
    else:
        constraints = None
    testsystem = LysozymeImplicit(constraints=constraints, implicitSolvent=app.OBC2)
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
    add_barostat(system)

    return topology, system, positions

temperature = simulation_parameters["temperature"]
n_samples = 5000
thinning_interval = 1000
burn_in_length = 5000
timestep = 0.75 * unit.femtosecond
from benchmark.testsystems.bookkeepers import EquilibriumSimulator

def construct_simulator(name, top, sys, pos):
    return EquilibriumSimulator(platform=configure_platform("CUDA"),
                         topology=top, system=sys, positions=pos,
                         temperature=temperature,
                         ghmc_timestep=timestep,
                         burn_in_length=burn_in_length, n_samples=n_samples,
                         thinning_interval=thinning_interval, name=name)

# DHFR
dhfr_constrained = construct_simulator("dhfr_constrained", *load_dhfr_explicit(constrained=True))
dhfr_unconstrained = construct_simulator("dhfr_unconstrained", *load_dhfr_explicit(constrained=False))


# T4 lysozyme
t4_constrained = construct_simulator("t4_constrained", *load_t4_implicit(constrained=True))
t4_unconstrained = construct_simulator("t4_unconstrained", *load_t4_implicit(constrained=False))

# constraint-coupled harmonic oscillators
top, sys, pos = load_constraint_coupled_harmonic_oscillators(constrained=True)
constraint_coupled_harmonic_oscillators = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=1000.0 * unit.femtosecond,
                                           burn_in_length=50, n_samples=10000,
                                           thinning_interval=10, name="constraint_coupled_harmonic_oscillators")
