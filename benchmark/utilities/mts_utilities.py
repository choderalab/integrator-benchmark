# Utilities for generating multi-timestep splittings
import itertools

import numpy as np
# To-do: add utilities for estimating the total number of force terms
from simtk import openmm as mm


def generate_solvent_solute_splitting_string(base_integrator="VRORV", K_p=1, K_r=3):
    """Generate string representing sequence of V0, V1, R, O steps, where force group 1
    is assumed to contain fast-changing, cheap-to-evaluate forces, and force group 0
    is assumed to contain slow-changing, expensive-to-evaluate forces.



    Currently only supports solvent-solute splittings of the VRORV (BAOAB / g-BAOAB)
    integrator, but it should be easy also to support splittings of the ABOBA integrator.

    Parameters
    -----------
    base_integrator: string
        Currently only supports VRORV
    K_p: int
        Number of times to evaluate force group 1 per timestep.
    K_r: int
        Number of inner-loop iterations

    Returns
    -------
    splitting_string: string
        Sequence of V0, V1, R, O steps, to be passed to LangevinSplittingIntegrator
    """
    assert (base_integrator == "VRORV" or base_integrator == "BAOAB")
    Rs = "R " * K_r
    inner_loop = "V1 " + Rs + "O " + Rs + "V1 "
    s = "V0 " + inner_loop * K_p + "V0"
    return s


def generate_mts_string(groups_of_fgs, R_before=True, n_R=1):
    """groups_of_fgs is a list of tuples, containing an iterable and an integer.
    E.g.
    [([3], 1),
    ([1,2], 10),
    ([0], 4)
    ]

    We execute the first group [3] once per timestep.
    We execute the second group [1,2] 10 times per timestep.
    We execute the third group [0] 4*10=40 times per timestep
    """

    def group_string(group):
        steps = ["R"] * n_R + ["V{}".format(i) for i in group]
        # print(steps)
        if R_before:
            single_iter = " ".join(steps)
        else:
            single_iter = " ".join(steps[::-1])

        # print(single_iter)
        return " " + single_iter + " "

    group, n_iter = groups_of_fgs[-1]
    inner_loop_string = group_string(group) * n_iter

    for (group, n_iter) in groups_of_fgs[::-1][1:]:
        if R_before:
            inner_loop_string = str(group_string(group) + inner_loop_string) * n_iter
        else:
            inner_loop_string = str(group_string(group) + inner_loop_string) * n_iter

    # make sure we don't have extraneous white spaces
    return " ".join(inner_loop_string.split())


def generate_from_ratios(bond_steps_per_angle_step=5,
                         angle_steps=7,
                         R_before=True,
                         ):
    groups_of_fgs = [([0, 1, 2, 3], 1),
                     ([0, 1, 2], angle_steps),
                     ([0], bond_steps_per_angle_step)
                     ]
    return generate_mts_string(groups_of_fgs, R_before)


def generate_gbaoab_solvent_solute_string(K_p=2, K_r=1, slow_group=[0], fast_group=[1]):
    """Following appendix of g-baoab paper.

    Let's say we want to evaluate force group 0 once per timestep and force group 1 twice per timestep.

    Using the default arguments above should give us

        V0 (V1 R^K_r O R^K_r V1)^K_p V0
        V0 V1 R O R V1 V1 R O R V1 V0

    Notes:
    * will perform K_p velocity randomizations per timestep
    *

    """
    Rs = ["R"] * K_r
    fast_forces = ["V{}".format(i) for i in fast_group]
    slow_forces = ["V{}".format(i) for i in slow_group]

    inner_loop_string = fast_forces + Rs + ["O"] + Rs + fast_forces
    return " ".join(slow_forces + inner_loop_string * K_p + slow_forces)


def generate_baoab_mts_string(groups, K_r=1):
    """Multi timestep generalization of the solvent-solute splitting scheme presented above...

    In the solvent-solute splitting, we have a "fast" group and a "slow" group.
    What if we have more than two groups?

    In the the straightforard generalization of the solvent-solute scheme, we do something like this:

    Accept groups, a list of 2-tuples, where each tuple contains an iterable of force group indices and
    an execution-frequency ratio.

    For example, groups=[([0], 1), ([1], 2), ([2], 2)] should be taken to mean:
        execute V1 twice as often as V0, and execute V2 twices as often as V1....

        To be concrete:

        If groups=[([0], 1), ([1], 2), ([2], 2)], K_r=1 we would have:

                V0 (V1 (V2 R^K_r O R^K_r V2)^2 V1)^2 V0

    """
    Rs = ["R"] * K_r

    ratios = [group[1] for group in groups]
    forces = [["V{}".format(i) for i in group[0]] for group in groups]

    inner_loop_string = forces[-1] + Rs + ["O"] + Rs + forces[-1]

    for i in range(len(ratios))[::-1][1:]:
        inner_loop_string = forces[i] + inner_loop_string * ratios[i - 1] + forces[i]

    return " ".join(inner_loop_string)


def generate_baoab_mts_string_from_ratios(bond_steps_per_angle_step=5, angle_steps=7):
    """Assuming there are just four groups.
    0: Bonds (Cheap)
    1,2: Angles and torsions (~4x more expensive than bonds)
    3: Nonbonded (~wayyyy more expensive than bonds)
    """
    return generate_baoab_mts_string([([3], 1), ([1, 2], angle_steps), ([0], angle_steps * bond_steps_per_angle_step)])


def generate_random_mts_string(n_updates_per_forcegroup, n_R_steps, n_O_steps):
    """
    n_updates_per_forcegroup is an array, where n_updates_per_forcegroup[i] is the number of times to call V0 in the
    """

    ingredients = []

    for i in range(len(n_updates_per_forcegroup)):
        for _ in range(n_updates_per_forcegroup[i]):
            ingredients.append("V{}".format(i))

    for _ in range(n_R_steps):
        ingredients.append("R")
    for _ in range(n_O_steps):
        ingredients.append("O")

    np.random.shuffle(ingredients)

    return " ".join(ingredients)


def generate_gbaoab_string(K_r=1):
    """K_r=1 --> 'V R O R V
    K_r=2 --> 'V R R O R R V'
    etc.
    """
    Rs = ["R"] * K_r
    return " ".join(["V"] + Rs + ["O"] + Rs + ["V"])


def generate_baoab_mts_string(groups, K_r=1):
    """Multi timestep generalization of the solvent-solute splitting scheme presented above...

    In the solvent-solute splitting, we have a "fast" group and a "slow" group.
    What if we have more than two groups?

    In the the straightforard generalization of the solvent-solute scheme, we do something like this:

    Accept groups, a list of 2-tuples, where each tuple contains an iterable of force group indices and
    an execution-frequency ratio.

    For example, groups=[([0], 1), ([1], 2), ([2], 2)] should be taken to mean:
        execute V1 twice as often as V0, and execute V2 twices as often as V1....

        To be concrete:

        If groups=[([0], 1), ([1], 2), ([2], 2)], K_r=1 we would have:

                V0 (V1 (V2 R^K_r O R^K_r V2)^2 V1)^2 V0

    """
    Rs = ["R"] * K_r

    ratios = [group[1] for group in groups]
    forces = [["V{}".format(i) for i in group[0]] for group in groups]

    inner_loop_string = (forces[-1] + Rs + ["O"] + Rs + forces[-1]) * ratios[-1]

    for i in range(len(ratios))[::-1][1:]:
        inner_loop_string = (forces[i] + inner_loop_string + forces[i]) * ratios[i]

    return " ".join(inner_loop_string)


def generate_baoab_mts_string_from_ratios(bond_steps_per_angle_step=5, angle_steps=7):
    """Assuming there are just four groups.
    0: Bonds (Cheap)
    1,2: Angles and torsions (let's say that's ~5x more expensive than bonds)
    3: Nonbonded (~wayyyy more expensive than bonded interactions)
    """
    return generate_baoab_mts_string([([3], 1), ([1, 2], angle_steps), ([0], bond_steps_per_angle_step)])


def condense_splitting(splitting_string):
    """Since some operators commute, we can simplify some splittings.

    Here, we replace repeated O steps or V{i} steps.

    Splitting is a list of steps.

    Examples:

        O V R V V V O should condense to
        O V R V O

        and O V O O V R V
        should condense to:
        O V R V

        since
    """

    # first split into chunks of either velocity or position updates
    # don't collapse position updates, do collapse velocity updates

    splitting = splitting_string.upper().split()

    equivalence_classes = {"R": {"R"}, "V": {"O", "V"}, "O": {"O", "V"}}

    current_chunk = [splitting[0]]
    collapsed = []

    def collapse_chunk(current_chunk):

        if current_chunk[0] == "R":
            return current_chunk
        else:
            return list(set(current_chunk))

    for i in range(1, len(splitting)):

        # if the next step in the splitting is
        if splitting[i][0] in equivalence_classes[splitting[i - 1][0]]:
            current_chunk.append(splitting[i])
        else:
            collapsed += collapse_chunk(current_chunk)
            current_chunk = [splitting[i]]

    collapsed = collapsed + collapse_chunk(current_chunk)

    collapsed_string = " ".join(collapsed)
    if len(collapsed) < len(splitting):
        print("Shortened the splitting from {} steps to {} steps ({} --> {})".format(
            len(splitting), len(collapsed), splitting_string, collapsed_string
        ))

    return collapsed_string


def generate_sequential_BAOAB_string(force_group_list, symmetric=True):
    """Generate BAOAB-like schemes that break up the "V R" step
    into multiple sequential updates

    E.g. force_group_list=(0,1,2), symmetric=True -->
        "V0 R V1 R V2 R O R V2 R V1 R V0"
    force_group_list=(0,1,2), symmetric=False -->
        "V0 R V1 R V2 R O V0 R V1 R V2 R"
    """

    VR = []
    for i in force_group_list:
        VR.append("V{}".format(i))
        VR.append("R")

    if symmetric:
        return " ".join(VR + ["O"] + VR[::-1])
    else:
        return " ".join(VR + ["O"] + VR)


def generate_all_BAOAB_permutation_strings(n_force_groups, symmetric=True):
    """Generate all of the permutations of range(n_force_groups)"""
    return [(perm, generate_sequential_BAOAB_string(perm, symmetric)) for perm in
            itertools.permutations(range(n_force_groups))]


# Utilities for modifying force groups
# TODO: Valence vs. nonbonded
# TODO: Short-range vs long-range
# TODO: Solute-solvent vs. solvent-solvent

# Kyle's function for splitting up the forces in a system

def valence_vs_nonbonded(system):
    pass


def short_range_vs_long_range(system):
    """Not sure the details of what this should do"""
    pass


def clone_nonbonded_parameters(nonbonded_force):
    """Creates a new (empty) nonbonded force with the same global parameters"""

    # call constructor
    new_force = nonbonded_force.__class__()

    # go through all of the setter and getter methods
    new_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
    new_force.setEwaldErrorTolerance(nonbonded_force.getEwaldErrorTolerance())
    # new_force.setExceptionParameters # this is per-particle-pair property
    new_force.setForceGroup(nonbonded_force.getForceGroup())
    new_force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
    new_force.setPMEParameters(*nonbonded_force.getPMEParameters())
    new_force.setReactionFieldDielectric(nonbonded_force.getReactionFieldDielectric())
    new_force.setReciprocalSpaceForceGroup(nonbonded_force.getReciprocalSpaceForceGroup())
    new_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
    new_force.setUseDispersionCorrection(nonbonded_force.getUseDispersionCorrection())
    new_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())

    # TODO: There's probably a cleaner, more Pythonic way to do this
    # (this was the most obvious way, but may not work in the future if the OpenMM API changes)

    return new_force


def split_nonbonded_force(nonbonded_force, atom_indices_0, atom_indices_1):
    """Split a nonbonded force into 3 forces: """

    num_particles = nonbonded_force.getNumParticles()

    # check that atom_indices_0 and atom_indices_1 form a partition of the particle set
    if set(atom_indices_0).union(set(atom_indices_1)) != set(range(num_particles)):
        raise (Exception("Some atoms are missing!"))
    if len(set(atom_indices_0).intersection(set(atom_indices_1))) > 0:
        raise (Exception("atom_indices_0 and atom_indices_1 overlap!"))

    # duplicate the global parameters of nonbonded force
    new_force_0 = clone_nonbonded_parameters(nonbonded_force)
    new_force_1 = clone_nonbonded_parameters(nonbonded_force)

    # add particles from each group
    mapping = dict()  # maps from particle index to force, particle number...
    for i in range(num_particles):
        if i in atom_indices_0:
            new_force = new_force_0
        else:
            new_force = new_force_1

        mapping[i] = (new_force, new_force.addParticle(*nonbonded_force.getParticleParameters(i)))

    # copy over all the interaction exceptions
    for exception_index in range(nonbonded_force.getNumExceptions()):
        exception_parameters = nonbonded_force.getExceptionParameters(exception_index)
        i, j = exception_parameters[:2]

        # if particles i and j are in the same new force...
        if mapping[i][0] == mapping[j][0]:
            the_force_theyre_both_in = mapping[i][0]
            # get the particle numbers the force they're both in knows them by
            mapped_i, mapped_j = mapping[i][1], mapping[j][1]
            the_force_theyre_both_in.addException(mapped_i, mapped_j, *exception_parameters[2:])

    # finally, add a new force containing only the interactions *between* atom_indices_0 and atom_indices_1
    new_force_01 = clone_nonbonded_parameters(nonbonded_force)

    # copy all particles over
    for i in range(num_particles):
        new_force_01.addParticle(*nonbonded_force.getParticleParameters(i))

    # copy all existing exceptions
    for exception_index in range(nonbonded_force.getNumExceptions()):
        exception_parameters = nonbonded_force.getExceptionParameters(exception_index)
        i, j = exception_parameters[:2]
        new_force_01.addException(i, j, *exception_parameters[2:])

    # add exceptions for all (i,j) where (i,j) both in atom_indices_0 or (i,j) both in atom_indices_1
    for i in atom_indices_0:
        for j in atom_indices_0:
            new_force_01.addException(i, j, 0, 0, 0, True)
    for i in atom_indices_1:
        for j in atom_indices_1:
            new_force_01.addException(i, j, 0, 0, 0, True)

    return new_force_0, new_force_1, new_force_01


def get_water_atom_indices(topology):
    """Get list of atom indices in "WAT" residues"""
    indices = []
    water_residues = [r for r in topology.residues() if r.name == "WAT"]
    for water_residue in water_residues:
        for a in water_residue.atoms():
            indices.append(a.index)
    return indices


def get_nonbonded_forces(system):
    return [force for force in system.getForces() if "Nonbonded" in force.__class__.__name__]


def check_system_and_topology_match(system, topology):
    """Check to make sure the particle indices of the system
    line up with the atom indices in the topology"""

    if system.getNumParticles() != topology.getNumAtoms():
        raise (Exception("They don't even have the same number of particles!"))


def solute_solvent(system, topology, solvent_solvent=0, others=1):
    """Splits all interactions into two force groups:
    - solvent-solvent interactions are one force group
    - all other interactions are another force group
    
    Internally, this is done by creating 3 forces, and assigning them to the two force groups:
    - solvent_solvent_force : solvent_solvent
    - solute_solute_force : others
    - solute_solvent_force : others
    
    Notes / possible surprises
    --------------------------
    - Leaves bonded solvent interactions in others!
    
    Parameters
    ----------
    system : openmm system
    topology : openmm topology
    solvent_solvent : int
    others : int

    Side-effects
    ------------
    modifies system

    Notes and references
    --------------------
    - Related discussion on OpenMM issue between John Chodera and Peter Eastman: 
      https://github.com/pandegroup/openmm/issues/1498
        I think this isn't quite the solvent-solute splitting algorithm described in the g-BAOAB paper.
        In this discussion, the positions of the water molecules are updated more frequently than the 
        In the paper, the *velocities* get
    """

    check_system_and_topology_match(system, topology)

    atom_indices_solvent = get_water_atom_indices(topology)
    atom_indices_solute = sorted(list(set(range(system.getNumParticles())).difference(set(atom_indices_solvent))))

    # by default, set force group to others
    for force in system.getForces():
        force.setForceGroup(others)

    for nonbonded_force in get_nonbonded_forces(system):
        solvent_solvent_force, solute_solute_force, solute_solvent_force = split_nonbonded_force(nonbonded_force,
                                                                                                 atom_indices_solvent,
                                                                                                 atom_indices_solute)

        # add these forces to the system
        system.addForce(solvent_solvent_force)
        system.addForce(solute_solute_force)
        system.addForce(solute_solvent_force)

        # assign them to the correct force group
        solvent_solvent_force.setForceGroup(solvent_solvent)
        solute_solute_force.setForceGroup(others)  # just to be safe
        solute_solvent_force.setForceGroup(others)  # just to be safe

        # TODO: Check that the system now has 3 times as many forces in it
        # TODO: Check that system energies / forces are the same before and after splitting


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
