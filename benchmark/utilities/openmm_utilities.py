import simtk.openmm as mm
from simtk import unit
from benchmark import simulation_parameters
import numpy as np
from copy import deepcopy

def get_total_energy(simulation):
    """Compute the kinetic energy + potential energy of the simulation."""
    state = simulation.context.getState(getEnergy=True)
    ke, pe = state.getKineticEnergy(), state.getPotentialEnergy()
    return ke + pe


def get_positions(simulation):
    """Get array of particle positions"""
    return simulation.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_velocities(simulation):
    """Get array of particle velocities"""
    return simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)


def set_positions(simulation, x):
    """Set particle positions"""
    simulation.context.setPositions(x)


def set_velocities(simulation, v):
    """Set particle velocities"""
    simulation.context.setVelocities(v)


# Utilities for modifying system masses
def is_hydrogen(atom):
    """Check if this atom is a hydrogen"""
    return atom.element.symbol == "H"


def get_hydrogens(topology):
    """Get the indices of hydrogen atoms"""

    atom_indices = []
    for (atom_index, atom) in enumerate(topology.atoms()):
        if is_hydrogen(atom):
            atom_indices.append(atom_index)
    return atom_indices

def set_hydrogen_mass(system, topology, h_mass=4.0):
    """Set the masses of all hydrogens in system to h_mass (amu)"""
    atom_indices = get_hydrogens(topology)
    for atom_index in atom_indices:
        system.setParticleMass(atom_index, h_mass)

def get_mass(system, atom_index):
    return system.getParticleMass(atom_index).value_in_unit(unit.amu)

def get_masses(system):
    masses = [system.getParticleMass(atom_index) for atom_index in range(system.getNumParticles())]
    m_unit = masses[0].unit

    return np.array([m.value_in_unit(m_unit) for m in masses]) * m_unit

def decrement_particle_masses(system, atom_indices, decrement):
    """Reduce the masses of all atoms in `atom_indices` by `decrement`"""
    for atom_index in atom_indices:
        current_mass = get_mass(system, atom_index)
        system.setParticleMass(atom_index, current_mass - decrement)


def scale_particle_masses(system, atom_indices, scale_factor):
    """Multiply the masses of all atoms in `atom_indices` by `scale_factor`"""
    for atom_index in atom_indices:
        current_mass = get_mass(system, atom_index)
        system.setParticleMass(atom_index, current_mass * scale_factor)


def get_sum_of_masses(system, atom_indices=None):
    """Get the sum of particle masses in the system"""
    if atom_indices == None:
        atom_indices = range(system.getNumParticles())
    return sum([get_mass(system, atom_index)
                for atom_index in atom_indices])

def set_masses_equal(system, mass, atom_indices=None):
    """Set the particle masses in the system"""
    if atom_indices == None:
        atom_indices = range(system.getNumParticles())

    for atom_index in atom_indices:
        system.setParticleMass(atom_index, mass)

def get_atoms_bonded_to_hydrogen(topology):
    """Get the indices of particles bonded to hydrogen"""

    atom_indices = []
    for (a, b) in topology.bonds():
        a_H, b_H = is_hydrogen(a), is_hydrogen(b)
        if a_H + b_H == 1:  # if exactly one of these is a hydrogen
            if is_hydrogen(a):
                atom_indices.append(b.index)
            else:
                atom_indices.append(a.index)
    return atom_indices

def repartition_hydrogen_mass_connected(topology, system, h_mass=4.0,
                                        mode="scale"  # or "decrement"
                                        ):
    """Set the mass of all hydrogens to h_mass. Reduce the mass of
    all atoms bonded to hydrogens, so that the total mass remains constant.

    Question: how should we do this, exactly?
    * Should we subtract the same mass from each bonded atom?
    * Should we proportionally reduce the mass of each bonded atom?
    """

    set_hydrogen_mass(system, topology, h_mass)
    atoms_bonded_to_H = get_atoms_bonded_to_hydrogen(topology)

    if mode == "scale":
        initial_mass_of_bonded_atoms = get_sum_of_masses(system, atoms_bonded_to_H)
        mass_to_remove_from_others = (h_mass - 1) * len(get_hydrogens(topology))
        scale_factor = 1 - (mass_to_remove_from_others / initial_mass_of_bonded_atoms)
        scale_particle_masses(system, atoms_bonded_to_H, scale_factor)

    elif mode == "decrement":
        delta_mass = h_mass - 1.0
        decrement_particle_masses(system, atoms_bonded_to_H, delta_mass)


def repartition_hydrogen_mass_all(topology, system, h_mass=4.0,
                                  mode="scale",  # or "decrement"
                                  ):
    """Set the mass of all hydrogens to h_mass. Reduce the mass of
    ({all other} or {connected}) atoms, so that the total mass remains constant.
    """
    initial_mass = get_sum_of_masses(system)
    hydrogens = get_hydrogens(topology)
    initial_hydrogen_mass = get_sum_of_masses(system, hydrogens)

    target_H_mass = h_mass * len(hydrogens)
    mass_to_remove_from_others = target_H_mass - initial_hydrogen_mass
    others = list(set(range(topology.getNumAtoms())) - set(hydrogens))

    if mode == "scale":
        scale_factor = 1 - (mass_to_remove_from_others / initial_mass)
        scale_particle_masses(system, others, scale_factor)

    elif mode == "decrement":
        decrement = mass_to_remove_from_others / len(others)
        decrement_particle_masses(system, others, decrement)

    set_hydrogen_mass(system, topology, h_mass)

def repartition_hydrogen_mass(topology, system, h_mass=4.0, mode="decrement", atoms="connected"):
    """Modify `system` by setting H mass and decreasing other atoms' mass
    
    Parameters
    ----------
    topology
    system
    h_mass 
    mode : string
        "decrement" : subtract the same mass from each other atom
        "scale" : proportionally reduce the mass of each other atom
    atoms : string
        "connected" : reduce mass of atoms bonded to H
        "all" : reduce mass of all non-H atoms
    """
    system = deepcopy(system)

    # check to make sure system mass is unchanged...
    pre_mass = get_sum_of_masses(system)

    if atoms == "connected":
        repartition = repartition_hydrogen_mass_all
    elif atoms == "all":
        repartition = repartition_hydrogen_mass_connected
    else:
        raise(NotImplementedError("`atoms` must be either `all` or `connected`!"))

    repartition(topology, system, h_mass, mode)

    post_mass = get_sum_of_masses(system)
    assert(pre_mass == post_mass)

# TODO: Reduce code duplication between repartition_hydrogen_mass_all and repartition_hydrogen_mass_connected]

# Utilities for modifying force groups
# TODO: Valence vs. nonbonded
# TODO: Short-range vs long-range
# TODO: Solute-solvent vs. solvent-solvent

# Kyle's function for splitting up the forces in a system
def guess_force_groups(system, nonbonded=1, fft=1, others=0, multipole=1):
    """Set NB short-range to 1 and long-range to 1, which is usually OK.
    This is useful for RESPA multiple timestep integrators.

    Reference
    ---------
    https://github.com/kyleabeauchamp/openmmtools/blob/hmc/openmmtools/hmc_integrators.py
    """
    for force in system.getForces():
        if isinstance(force, mm.openmm.NonbondedForce):
            force.setForceGroup(nonbonded)
            force.setReciprocalSpaceForceGroup(fft)
        elif isinstance(force, mm.openmm.CustomGBForce):
            force.setForceGroup(nonbonded)
        elif isinstance(force, mm.openmm.GBSAOBCForce):
            force.setForceGroup(nonbonded)
        elif isinstance(force, mm.AmoebaMultipoleForce):
            force.setForceGroup(multipole)
        elif isinstance(force, mm.AmoebaVdwForce):
            force.setForceGroup(nonbonded)
        else:
            force.setForceGroup(others)

def remove_barostat(system):
    """Remove any force with "Barostat" in the name"""
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if "Barostat" in force.__class__.__name__:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        print('   Removing %s' % system.getForce(force_index).__class__.__name__)
        system.removeForce(force_index)

def add_barostat(system):
    """Add Monte Carlo barostat"""
    system.addForce(mm.MonteCarloBarostat(simulation_parameters["pressure"],
                                          simulation_parameters["temperature"]))

def keep_only_some_forces(system, extra_forces_to_keep=[]):
    """Remove unwanted forces, e.g. center-of-mass motion removal"""
    forces_to_keep = extra_forces_to_keep + ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                                             "NonbondedForce"]
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ not in forces_to_keep:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        print('   Removing %s' % system.getForce(force_index).__class__.__name__)
        system.removeForce(force_index)
