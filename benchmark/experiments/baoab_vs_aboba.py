from benchmark.testsystems import alanine_unconstrained, NonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator


BAOAB = NonequilibriumSimulator(alanine_unconstrained,
                                LangevinSplittingIntegrator("V R O R V"))

ABOBA = NonequilibriumSimulator(alanine_unconstrained,
                                LangevinSplittingIntegrator("R V O V R"))

