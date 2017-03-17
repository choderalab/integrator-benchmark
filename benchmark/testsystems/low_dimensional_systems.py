import numpy as np
from openmmtools.testsystems import CustomExternalForcesTestSystem, AlanineDipeptideVacuum, WaterBox, AlanineDipeptideExplicit, SrcImplicit
from simtk.openmm import app
from simtk import unit
from configuration import configure_platform
from benchmark.utilities import keep_only_some_forces

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