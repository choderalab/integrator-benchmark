import numpy as np
from openmmtools.testsystems import CustomExternalForcesTestSystem, AlanineDipeptideVacuum, WaterBox, AlanineDipeptideExplicit, SrcImplicit
from simtk.openmm import app
from simtk import unit
from configuration import configure_platform

n_particles = 500
def load_harmonic_oscillator(**args):
    """Load 3D harmonic oscillator"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^2 + {k}*y^2 + {k}*z^2".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions

def load_quartic_potential(**args):
    """Load 3D quartic potential"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions

def load_mts_test(**args):
    """
    n_particles : int
        number of identical, independent particles to add
        this is just an efficiency thing -- can simulate many replicates in parallel, instead of spending
        the openmm overhead to get a single replicate at a time

        to-do: maybe have the customintegrator keep track of the shadow work of each DOF separately?
            that way, our estimates / uncertainty estimates aren't messed up (e.g. it doesn't look like
            we have just 1,000 samples, when in fact we have 500,000 samples)
        to-do: maybe have the customintegrator keep track of the shadow work due to each force group separately?
    """
    ks = [100.0, 400.0] # stiffness of each force group term
    # force group term 0 will be evaluated most slowly, etc...
    testsystem = CustomExternalForcesTestSystem(energy_expressions=["{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=k) for k in ks],
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions


def load_waterbox(constrained=True):
    """Load WaterBox test system with non-default PME cutoff and error tolerance... """
    # print acceptance rates
    # use larger amount of equilibration
    # also use NPT for equilibration
    # make sure there's a barostat
    # use NPT between GHMC samples
    # change PME tolerance to 1e-5 or tighter
    # change cutoff to 10A
    # LJ turn on switch if not on by default


    testsystem = WaterBox(constrained=constrained, ewaldErrorTolerance=1e-5, cutoff=10*unit.angstroms)
    (topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions
    positions = np.load("waterbox.npy") # load equilibrated configuration saved beforehand
    return topology, system, positions

def keep_only_some_forces(system, extra_forces_to_keep=[]):
    """Remove unwanted forces, e.g. center-of-mass motion removal"""
    forces_to_keep = extra_forces_to_keep + ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce"]
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ not in forces_to_keep:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        print('   Removing %s' % system.getForce(force_index).__class__.__name__)
        system.removeForce(force_index)

def load_alanine(constrained=True):
    """Load AlanineDipeptide vacuum, optionally with hydrogen bonds constrained"""
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = AlanineDipeptideVacuum(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    keep_only_some_forces(system)

    return topology, system, positions

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

# TODO: Add Waterbox, AlanineExplicit EquilibriumSimulators