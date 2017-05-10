import numpy as np
from benchmark.testsystems.alanine_dipeptide import load_alanine
from benchmark.utilities.openmm_utilities import get_sum_of_masses, get_masses, repartition_hydrogen_mass, \
    get_hydrogens, repartition_hydrogen_mass_amber
from simtk import unit


def check_hmr_conserves_mass(mode, atoms, h_mass, topology, system):
    """Check that sum_i m_i before HMR = sum_i m_i after HMR"""
    pre_hmr_mass = get_sum_of_masses(system)
    hmr_system = repartition_hydrogen_mass(topology, system, h_mass, mode, atoms)
    post_hmr_mass = get_sum_of_masses(hmr_system)

    if not np.isclose(post_hmr_mass, pre_hmr_mass):
        raise Exception(
            "HMR failed to conserve total system mass!\n\tPre-HMR mass: {:.3f}\n\tPost-HMR mass: {:.3f}\n\tmode: {}\n\tatoms: {}\n\th_mass: {}".format(
                pre_hmr_mass, post_hmr_mass, mode, atoms, h_mass))


def check_hmr_leaves_particle_mass_positive(mode, atoms, h_mass, topology, system):
    """Check that min(m_i) after HMR >= 0"""
    pre_hmr_masses = get_masses(system)
    hmr_system = repartition_hydrogen_mass(topology, system, h_mass, mode, atoms)
    post_hmr_masses = get_masses(hmr_system)

    if min(post_hmr_masses).value_in_unit(unit.amu) <= 0:
        message = "HMR set some masses <= 0!"
        if min(pre_hmr_masses).value_in_unit(unit.amu) <= 0:
            message = message + " (also, some masses *before* the test were <= 0!)"
        raise Exception(message + "\n\tmode: {}\n\tatoms: {}\n\th_mass: {}".format(mode, atoms, h_mass))


def check_hmr_sets_h_mass_appropriately(mode, atoms, h_mass, topology, system):
    """Check that the masses of all hydrogen atoms after HMR = h_mass"""

    hmr_system = repartition_hydrogen_mass(topology, system, h_mass, mode, atoms)
    h_masses = [hmr_system.getParticleMass(atom_index) for atom_index in get_hydrogens(topology)]

    for h in h_masses:
        if h != h_mass * unit.amu:
            raise Exception(
                "HMR didn't set all hydrogen masses to `h_mass`!\n\tmode: {}\n\tatoms: {}\n\th_mass: {}".format(mode,
                                                                                                                atoms,
                                                                                                                h_mass))


def check_amber_hmr_leaves_particle_mass_positive(topology, system):
    """Check that min(m_i) after AMBER HMR >= 0"""
    hmr_system = repartition_hydrogen_mass_amber(topology, system)
    post_hmr_masses = get_masses(hmr_system)

    if min(post_hmr_masses).value_in_unit(unit.amu) <= 0:
        raise Exception("HMR (AMBER scheme) set some masses <= 0!")


def check_amber_hmr_conserves_mass(topology, system):
    """Check that sum_i m_i before AMBER HMR = sum_i m_i after AMBER HMR"""
    pre_hmr_mass = get_sum_of_masses(system)
    hmr_system = repartition_hydrogen_mass_amber(topology, system)
    post_hmr_mass = get_sum_of_masses(hmr_system)

    if not np.isclose(post_hmr_mass, pre_hmr_mass):
        raise Exception(
            "HMR (AMBER scheme) failed to conserve total system mass!\n\tPre-HMR mass: {:.3f}\n\tPost-HMR mass: {:.3f}".format(
                pre_hmr_mass, post_hmr_mass))


def check_amber_hmr_sets_h_mass_appropriately(topology, system):
    """Check that the masses of all hydrogen atoms after HMR = h_mass"""
    hydrogens = get_hydrogens(topology)
    pre_hmr_h_mass = get_sum_of_masses(system, hydrogens)
    hmr_system = repartition_hydrogen_mass_amber(topology, system)
    post_hmr_h_mass = get_sum_of_masses(hmr_system, hydrogens)

    computed_scale_factor = post_hmr_h_mass / pre_hmr_h_mass
    target_scale_factor = 3
    if not np.isclose(computed_scale_factor, target_scale_factor):
        raise Exception(
            "HMR (AMBER scheme) didn't scale hydrogen mass by {}x!\n\t(post-HMR H mass) / (pre-HMR H mass): {:.3f}".format(
                target_scale_factor, computed_scale_factor))


def test_hmr():
    topology, system, _ = load_alanine()
    for mode in ["decrement", "scale"]:
        for atoms in ["connected", "all"]:
            for h_mass in [0.1, 1.0, 4.0]:
                yield check_hmr_conserves_mass, mode, atoms, h_mass, topology, system
                yield check_hmr_leaves_particle_mass_positive, mode, atoms, h_mass, topology, system
                yield check_hmr_sets_h_mass_appropriately, mode, atoms, h_mass, topology, system

    yield check_amber_hmr_leaves_particle_mass_positive, topology, system
    yield check_amber_hmr_conserves_mass, topology, system
    yield check_amber_hmr_sets_h_mass_appropriately, topology, system
