from benchmark.testsystems.alanine_dipeptide import load_alanine
from benchmark.utilities.openmm_utilities import get_sum_of_masses, repartition_hydrogen_mass
from unittest import TestCase

def check_hmr_conserves_mass(mode, atoms, h_mass, topology, system):
    pre_hmr_mass = get_sum_of_masses(system)
    hmr_system = repartition_hydrogen_mass(topology, system, h_mass, mode, atoms)
    post_hmr_mass = get_sum_of_masses(hmr_system)
    if post_hmr_mass != pre_hmr_mass:
        raise Exception("HMR failed to conserve total system mass!\n\tmode: {}\n\tatoms: {}\n\th_mass: {}".format(mode, atoms, h_mass))


def test_hmr_preserves_total_mass():
    topology, system, _ = load_alanine()
    for mode in ["decrement", "scale"]:
        for atoms in ["connected", "all"]:
            for h_mass in [0.1, 1.0, 4.0, 8.0]:
                check_hmr_conserves_mass.description = ("Testing that repartitioning hydrogen mass conserves total system mass\n\tmode: {}\n\tatoms: {}\n\th_mass: {}".format(mode, atoms, h_mass))
                yield check_hmr_conserves_mass, mode, atoms, h_mass, topology, system
