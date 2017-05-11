# Utilities for generating multi-timestep splittings
import numpy as np
import itertools

# To-do: add utilities for estimating the total number of force terms
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
    assert(base_integrator == "VRORV" or base_integrator == "BAOAB")
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

    equivalence_classes = {"R":{"R"}, "V":{"O", "V"}, "O":{"O", "V"}}

    current_chunk = [splitting[0]]
    collapsed = []

    def collapse_chunk(current_chunk):

        if current_chunk[0] == "R":
            return current_chunk
        else:
            return list(set(current_chunk))

    for i in range(1, len(splitting)):

        # if the next step in the splitting is
        if splitting[i][0] in equivalence_classes[splitting[i-1][0]]:
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
    """Generate all of the permutations of range(n_force_groups), and evaluate their
    acceptance rates
    """
    return [(perm, generate_sequential_BAOAB_string(perm, symmetric)) for perm in itertools.permutations(range(n_force_groups))]
