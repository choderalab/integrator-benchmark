import numpy as np
from openmmtools.testsystems import HarmonicOscillator, PowerOscillator, \
    AlanineDipeptideVacuum, WaterBox
from simtk.openmm import app

from utils import configure_platform
from simtk import unit
import simtk.openmm as mm

def load_harmonic_oscillator(**args):
    """Load 3D harmonic oscillator"""
    testsystem = HarmonicOscillator()
    return testsystem.topology, testsystem.system, testsystem.positions

def load_quartic_potential(**args):
    """Load 3D quartic potential"""
    testsystem = PowerOscillator(b=4.0)
    return testsystem.topology, testsystem.system, testsystem.positions

def load_waterbox(constrained=True):
    """Load WaterBox test system with non-default PME cutoff and error tolerance... """
    testsystem = WaterBox(constrained=constrained, ewaldErrorTolerance=1e-5, cutoff=10*unit.angstroms)
    (topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions
    positions = np.load("waterbox.npy") # load equilibrated configuration saved beforehand
    return topology, system, positions


def load_alanine(constrained=True):
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = AlanineDipeptideVacuum(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    index_of_CMMotionRemover = -1
    for i in range(system.getNumForces()):
        if type(system.getForce(i)) == mm.CMMotionRemover:
            index_of_CMMotionRemover = i
    if index_of_CMMotionRemover != -1:
        system.removeForce(i)

    return topology, system, positions


simple_params = {
    "platform": configure_platform("Reference"),
    "burn_in_length": 1000,
    "n_samples": 1000,
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

system_params = {
    "harmonic_oscillator": harmonic_oscillator_params,
    "quartic_potential": quartic_params,
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
        "protocol_length": 50,
        "constrained_timestep": 2.5*unit.femtosecond,
        "unconstrained_timestep": 2.0*unit.femtosecond,
        "temperature": 298.0 * unit.kelvin,
        "collision_rate": 91 / unit.picoseconds,
    }
}