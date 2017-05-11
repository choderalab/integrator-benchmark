# Quick check of several HMR schemes against several test systems

from functools import partial

from benchmark.utilities.openmm_utilities import get_masses, get_vibration_timescales, \
    ratio_of_largest_and_shortest_timescale
from benchmark.utilities.openmm_utilities import repartition_hydrogen_mass, repartition_hydrogen_mass_amber
from openmmtools.testsystems import AlanineDipeptideVacuum, AlanineDipeptideExplicit, \
    DHFRExplicit

testsystems = [AlanineDipeptideVacuum(constraints=None),
               AlanineDipeptideExplicit(constraints=None, rigid_water=False),
               DHFRExplicit(constraints=None, rigid_water=False)
               ]

hmr_schemes = {
    "AMBER (2x)": partial(repartition_hydrogen_mass_amber, scale_factor=2),
    "AMBER (3x)": partial(repartition_hydrogen_mass_amber, scale_factor=3),
    "AMBER (4x)": partial(repartition_hydrogen_mass_amber, scale_factor=4),
}
for mode in ["decrement", "scale"]:
    for atoms in ["connected", "all"]:
        for h_mass in [2.0, 3.0, 4.0]:
            hmr_schemes["{} ({}) H mass={}".format(mode, atoms, h_mass)] = partial(repartition_hydrogen_mass, mode=mode,
                                                                                   atoms=atoms, h_mass=h_mass)
max_name_length = max([len(name) for name in hmr_schemes])


def format_scheme_name(scheme_name):
    return scheme_name.ljust(max_name_length + 2)


# TODO: Add a scheme that just numerically optimizes masses to equalize vibration timescales
# subject to bounds (hmr_mass[i] > 0) and constraints (sum(hmr_mass[i]) = sum(pre_mass[i])

def strip_unit(x):
    """Strip unit from a simtk unit'd quantity"""
    # TODO: Remove this when I find what the problem is...
    try:
        return x.value_in_unit(x.unit)
    except:
        return x


for testsystem in testsystems:
    print(testsystem.__class__.__name__)
    topology, system = testsystem.topology, testsystem.system
    default_masses = get_masses(system)
    default_timescales = get_vibration_timescales(system, default_masses)
    default_t_ratio = ratio_of_largest_and_shortest_timescale(default_timescales)
    default_name_string = format_scheme_name("Default max(t_i) / min(t_i)")
    print("\t{}: {:.3f}".format(default_name_string, default_t_ratio))

    for name in sorted(hmr_schemes.keys()):
        scheme = hmr_schemes[name]
        hmr_system = scheme(topology, system)
        hmr_masses = get_masses(hmr_system)
        timescales = get_vibration_timescales(hmr_system, hmr_masses)
        timescales = [strip_unit(t) for t in timescales]  # TODO: Remove when intermittent unit error is debugged
        t_ratio = ratio_of_largest_and_shortest_timescale(timescales)
        name_string = format_scheme_name(name)
        print(
            "\t{}: {:.3f}\t(Improvement over default: {:.3f}x)".format(name_string, t_ratio, default_t_ratio / t_ratio))

# TODO: Accumulate and present statistics here: which scheme best equalizes vibrational timescales?
