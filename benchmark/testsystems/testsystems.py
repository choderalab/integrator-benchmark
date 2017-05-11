import numpy as np
from openmmtools.testsystems import SrcImplicit, DHFRExplicit
from simtk.openmm import app
from simtk import unit
from benchmark.testsystems.configuration import configure_platform
from benchmark.utilities import keep_only_some_forces
from benchmark import simulation_parameters
from benchmark.utilities import add_barostat
temperature = simulation_parameters["temperature"]


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
    add_barostat(system)

    return topology, system, positions

temperature = simulation_parameters["temperature"]
n_samples = 100
from benchmark.testsystems.bookkeepers import EquilibriumSimulator
top, sys, pos = load_dhfr_explicit(constrained=True)
dhfr_constrained = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=0.5 * unit.femtosecond,
                                           burn_in_length=500, n_samples=n_samples,
                                           thinning_interval=10, name="dhfr_constrained")

top, sys, pos = load_dhfr_explicit(constrained=False)
dhfr_unconstrained = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=0.5 * unit.femtosecond,
                                           burn_in_length=500, n_samples=n_samples,
                                           thinning_interval=10, name="dhfr_unconstrained")

top, sys, pos = load_src_implicit(constrained=True)
src_constrained = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=0.5 * unit.femtosecond,
                                           burn_in_length=500, n_samples=n_samples,
                                           thinning_interval=10, name="src_constrained")

top, sys, pos = load_src_implicit(constrained=False)
src_unconstrained = EquilibriumSimulator(platform=configure_platform("OpenCL"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=0.5 * unit.femtosecond,
                                           burn_in_length=500, n_samples=n_samples,
                                           thinning_interval=10, name="src_unconstrained")
