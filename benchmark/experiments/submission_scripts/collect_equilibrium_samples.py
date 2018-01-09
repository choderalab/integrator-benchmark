import sys

from benchmark.testsystems import dhfr_constrained, dhfr_unconstrained, \
    waterbox_constrained, flexible_waterbox, t4_constrained, t4_unconstrained, \
    alanine_constrained, alanine_unconstrained, src_constrained
from benchmark.testsystems.alanine_dipeptide import solvated_alanine_constrained, solvated_alanine_unconstrained
from benchmark.testsystems.testsystems import dhfr_reaction_field

systems = [dhfr_constrained, dhfr_unconstrained, src_constrained, \
           waterbox_constrained, flexible_waterbox, t4_constrained, t4_unconstrained, \
           alanine_constrained, alanine_unconstrained, solvated_alanine_constrained, solvated_alanine_unconstrained,
           dhfr_reaction_field]

for i in range(len(systems)):
    print(i, systems[i])

if __name__ == "__main__":
    systems[int(sys.argv[1]) - 1].sample_x_from_equilibrium()
