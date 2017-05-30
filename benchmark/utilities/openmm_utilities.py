from copy import deepcopy

import numpy as np
import simtk.openmm as mm
from benchmark import simulation_parameters
from simtk import unit

def get_potential_energy(simulation):
    return simulation.context.getState(getEnergy=True).getPotentialEnergy()


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
    """Get mass of a single particle"""
    return system.getParticleMass(atom_index).value_in_unit(unit.amu)


def get_masses(system):
    """Get array of masses of all particles"""
    masses = [system.getParticleMass(atom_index) for atom_index in range(system.getNumParticles())]
    m_unit = masses[0].unit

    return np.array([m.value_in_unit(m_unit) for m in masses]) * m_unit


def decrement_particle_masses(system, atom_indices, decrement):
    """Reduce the masses of all atoms in `atom_indices` by `decrement`"""
    for atom_index in atom_indices:
        current_mass = get_mass(system, atom_index)
        target_mass = current_mass - decrement
        if target_mass <= 0:
            raise (RuntimeError("Trying to remove too much mass!"))
        system.setParticleMass(atom_index, target_mass)


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
    """Get the indices of particles bonded to hydrogen
    
    Returns
    -------
    atom_indices : list of ints
        Will include repeated entries, if an atom is bonded to
        more than one hydrogen
    """

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
                                        mode="decrement"  # or "scale"
                                        ):
    """Set the mass of all hydrogens to h_mass. Reduce the mass of
    all atoms bonded to hydrogens, so that the total mass remains constant.
    """

    others = list(set(get_atoms_bonded_to_hydrogen(topology)))
    hydrogens = get_hydrogens(topology)
    initial_h_mass = get_sum_of_masses(system, hydrogens) / len(hydrogens)
    initial_mass_of_others = get_sum_of_masses(system, others)
    mass_to_remove_from_others = (h_mass - initial_h_mass) * len(hydrogens)

    if mode == "scale":
        scale_factor = (initial_mass_of_others - mass_to_remove_from_others) / initial_mass_of_others
        if scale_factor <= 0:
            raise (RuntimeError("h_mass is too large! Can't remove this much mass from bonded atoms..."))

        scale_particle_masses(system, others, scale_factor)

    elif mode == "decrement":
        decrement = mass_to_remove_from_others / len(others)
        decrement_particle_masses(system, others, decrement)
    set_hydrogen_mass(system, topology, h_mass)


def repartition_hydrogen_mass_all(topology, system, h_mass=4.0,
                                  mode="decrement",  # or "scale"
                                  ):
    """Set the mass of all hydrogens to h_mass. Reduce the mass of
     all other atoms, so that the total mass remains constant."""
    hydrogens = get_hydrogens(topology)
    initial_hydrogen_mass = get_sum_of_masses(system, hydrogens)

    target_H_mass = h_mass * len(hydrogens)
    mass_to_remove_from_others = target_H_mass - initial_hydrogen_mass
    others = list(set(range(topology.getNumAtoms())) - set(hydrogens))
    initial_mass_of_others = get_sum_of_masses(system, others)

    if mode == "scale":
        scale_factor = (initial_mass_of_others - mass_to_remove_from_others) / initial_mass_of_others
        scale_particle_masses(system, others, scale_factor)

    elif mode == "decrement":
        decrement = mass_to_remove_from_others / len(others)
        decrement_particle_masses(system, others, decrement)

    set_hydrogen_mass(system, topology, h_mass)


def repartition_hydrogen_mass(topology, system, h_mass=4.0, mode="decrement", atoms="connected"):
    """Return a modified copy of `system`, setting hydrogen mass and decreasing other atoms' mass.
    
    If mode == "decrement", subtract a constant from each other atom's mass
    If mode == "scale", multiply each other atom's mass by a constant
    If atoms == "connected", decrease the masses of only atoms bonded to hydrogen
    If atoms == "all", decrease the masses of all non-hydrogen atoms
    
    Parameters
    ----------
    topology : openmm topology
    system : openmm system
    h_mass : float
        target hydrogen mass, in amu
    mode : string
        "decrement" : subtract the same mass from each other atom
        "scale" : proportionally reduce the mass of each other atom
    atoms : string
        "connected" : reduce mass of atoms bonded to H
        "all" : reduce mass of all non-H atoms
    
    Returns
    -------
    hmr_system : openmm system
    """
    hmr_system = deepcopy(system)

    if atoms == "connected":
        repartition = repartition_hydrogen_mass_connected
    elif atoms == "all":
        repartition = repartition_hydrogen_mass_all
    else:
        raise (NotImplementedError("`atoms` must be either `all` or `connected`!"))

    repartition(topology, hmr_system, h_mass, mode)

    return hmr_system


def repartition_hydrogen_mass_amber(topology, system, scale_factor=3):
    """Scale up hydrogen mass, subtract added mass from bonded heavy atoms
    
    Algorithm
    ---------
    1. Multiply the masses of all hydrogens by 3
    2. Subtract the added mass from the bonded heavy atoms
        For example, if we have an atom bonded to one hydrogen,
        we subtract 1 * (scale_factor - 1) * initial_hydrogen_mass
        
        If we have an atom bonded to three hydrogens,
        we subtract 3 * (scale_factor - 1) * initial_hydrogen_mass
    
    Parameters
    ----------
    topology
    system
    scale_factor

    Returns
    -------
    hmr_system

    References
    ----------
    Long-Time-Step Molecular Dynamics through Hydrogen Mass Repartitioning
    [Hopkins, Grand, Walker, Roitberg, 2015, JCTC]
    http://pubs.acs.org/doi/abs/10.1021/ct5010406
    """
    hmr_system = deepcopy(system)

    # scale hydrogen mass by 3x, keeping track of how much mass was added to each
    hydrogens = get_hydrogens(topology)
    initial_h_masses = [get_mass(system, h) for h in hydrogens]
    if len(set(initial_h_masses)) > 1:
        raise(NotImplementedError("Initial hydrogen masses aren't all equal. "
                                  "Implementation currently assumes all hydrogen masses are initially equal."))
        # TODO: Relax this assumption

    scale_particle_masses(hmr_system, hydrogens, scale_factor)
    mass_added_to_each_hydrogen = get_mass(hmr_system, hydrogens[0]) - get_mass(system, hydrogens[0])

    # for each heavy-atom-hydrogen bond, subtract that amount of mass from the heavy atom
    for heavy_atom in get_atoms_bonded_to_hydrogen(topology):
        decrement_particle_masses(hmr_system, [heavy_atom], mass_added_to_each_hydrogen)

    return hmr_system

# Heuristic evaluation of HMR scheme: does it equalize vibrational timescales,
# assuming all bonds are independent?
def get_vibration_timescales(system, masses):
    """Get list of bond vibration timescales"""
    bonds = get_harmonic_bonds(system)
    timescales = []
    for (i,j,_,k) in bonds:
        timescales.append(bond_vibration_timescale(masses[i], masses[j], k))
    return timescales

def get_harmonic_bonds(system):
    """Get a list of all harmonic bonds in the system"""
    bonds = []
    for f in system.getForces():
        if "HarmonicBond" in str(f.__class__):
            for i in range(f.getNumBonds()):
                bonds.append(f.getBondParameters(i))
    return bonds

def bond_vibration_timescale(m1, m2, k):
    """Get period of two masses on a spring"""
    m = reduced_mass(m1, m2)
    return np.sqrt(k / m)

def reduced_mass(m1, m2):
    return m1 * m2 / (m1 + m2)

def difference_between_largest_and_shortest_timescale(timescales):
    return max(timescales) - min(timescales)

def ratio_of_largest_and_shortest_timescale(timescales):
    return max(timescales) / min(timescales)

# TODO: Reduce code duplication between repartition_hydrogen_mass_all and repartition_hydrogen_mass_connected]


def remove_barostat(system):
    """Remove any force with "Barostat" in the name"""
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if "Barostat" in force.__class__.__name__:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        force_name = system.getForce(force_index).__class__.__name__
        print("\tRemoving {}".format(force_name))
        system.removeForce(force_index)

def remove_center_of_mass_motion_remover(system):
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if "CMMotionRemover" in force.__class__.__name__:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        force_name = system.getForce(force_index).__class__.__name__
        print("\tRemoving {}".format(force_name))
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
        force_name = system.getForce(force_index).__class__.__name__
        print("\tRemoving {}".format(force_name))
        system.removeForce(force_index)
