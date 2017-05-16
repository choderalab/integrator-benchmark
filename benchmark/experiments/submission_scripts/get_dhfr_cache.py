# Tries to get a sample from the sample cache.
# This triggers populating the cache, if it doesn't already exist.
from benchmark.testsystems import dhfr_constrained
dhfr_constrained.sample_x_from_equilibrium()
