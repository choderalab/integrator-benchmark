from benchmark.utilities.mts_utilities import solute_solvent
import openmmtools
from simtk.openmm import app
import numpy as np

def check_solute_solvent_splitting_leaves_forces_unmodified(testsystem):
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    integrator = openmmtools.integrators.GHMCIntegrator()

    simulation = app.Simulation(top, sys, integrator)
    simulation.context.setPositions(pos)

    pre_forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)

    del(simulation)
    del(integrator)

    solute_solvent(sys, top)

    integrator = openmmtools.integrators.GHMCIntegrator()
    simulation = app.Simulation(top, sys, integrator)
    simulation.context.setPositions(pos)

    post_forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)

    if not np.allclose(pre_forces, post_forces):
        raise(Exception("Forces modified by solvent-solute splitting!"))

def test_solute_solvent_splitting():
    testsystem = openmmtools.testsystems.AlanineDipeptideExplicit()
    yield check_solute_solvent_splitting_leaves_forces_unmodified, testsystem
