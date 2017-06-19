# For each system, compute the configuration-space error to high accuracy.

import sys

import numpy as np
from simtk import unit

from benchmark.evaluation import estimate_nonequilibrium_free_energy
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems import dhfr_constrained, dhfr_unconstrained, \
    waterbox_constrained, flexible_waterbox, t4_constrained, t4_unconstrained, \
    alanine_constrained, alanine_unconstrained, src_constrained

systems = [None, dhfr_constrained, dhfr_unconstrained, src_constrained, \
           waterbox_constrained, flexible_waterbox, t4_constrained, t4_unconstrained, \
           alanine_constrained, alanine_unconstrained]

for i in range(len(systems)):
    print(i, systems[i])

if __name__ == "__main__":
    reference_integrator = LangevinSplittingIntegrator("O V R V O", collision_rate=1.0 / unit.picosecond,
                                                       timestep=2.0 * unit.femtosecond)

    noneq_sim = NonequilibriumSimulator(systems[int(sys.argv[1])], reference_integrator)

    n_protocol_samples = 1000
    protocol_length = 2000

    W_F, W_R = noneq_sim.collect_protocol_samples(n_protocol_samples, protocol_length)
    DeltaF_neq, sq_unc = estimate_nonequilibrium_free_energy(W_F, W_R)

    print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(sq_unc)))
