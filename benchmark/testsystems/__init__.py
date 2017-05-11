from .alanine_dipeptide import alanine_constrained, alanine_unconstrained
from .alanine_dipeptide import solvated_alanine_constrained, solvated_alanine_unconstrained
from .waterbox import flexible_waterbox, waterbox_constrained
from .coupled_power_oscillators import coupled_power_oscillators
from .low_dimensional_systems import double_well, quartic, NumbaNonequilibriumSimulator
from .bookkeepers import EquilibriumSimulator, NonequilibriumSimulator
from .testsystems import dhfr_constrained, dhfr_unconstrained, src_constrained

from .alanine_dipeptide import load_alanine, load_solvated_alanine
from .testsystems import load_src_implicit, load_dhfr_explicit, load_src_vacuum
from .waterbox import load_waterbox

system_loaders = {"alanine": load_alanine,
                  "solvated_alanine": load_solvated_alanine,
                  "src_vacuum": load_src_vacuum,
                  "src_implicit": load_src_implicit,
                  "dhfr_explicit": load_dhfr_explicit}

__all__ = ["alanine_constrained", "alanine_unconstrained", "solvated_alanine_unconstrained",
           "flexible_waterbox", "waterbox_constrained", "coupled_power_oscillators",
           "dhfr_constrained", "dhfr_unconstrained", "src_constrained",
           "EquilibriumSimulator", "NonequilibriumSimulator", "double_well", "quartic", "NumbaNonequilibriumSimulator", "system_loaders"]
