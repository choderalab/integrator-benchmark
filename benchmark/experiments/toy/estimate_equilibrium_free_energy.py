# Get precise estimates for the equilibrium free energy, by
# numerical quadrature.
# We will use this for:
# * Validating the non-parametric estimator of entropy and free energy
# * Computing nonequilibrium free energy differences

from quartic_simple import potential, kinetic_energy, log_p, log_v_density, beta
import numpy as np

x = np.linspace(-10, 10, 10000)
p_x = np.exp(log_p(x))

entropy_integrand = p_x * log_p(x)
entropy_x = - np.trapz(entropy_integrand, x)

potential_integrand = p_x * potential(x)
average_potential = np.trapz(potential_integrand, x)

print("<U> = {}".format(average_potential))
print("S_config = {}".format(entropy_x))

v = np.linspace(-10, 10, 10000)
p_v = np.exp(log_v_density(v))

entropy_integrand = p_v * log_v_density(v)
entropy_v = - np.trapz(entropy_integrand, v)

kinetic_integrand = p_v * kinetic_energy(v)
average_kinetic_energy = np.trapz(kinetic_integrand, v)
print("<KE> = {}".format(average_kinetic_energy))
print("S_momenta = {}".format(entropy_v))

# Since the marginals are independent, the total entropy is the
# sum of the marginal entropies
entropy = entropy_x + entropy_v
average_energy = average_potential + average_kinetic_energy
print("<E> = {}".format(average_energy))
print("S = {}".format(entropy))

print("Equilibrium free energy: {}".format(average_energy - entropy / beta))
print("x-marginal equilibrium free energy: {}".format(average_potential - entropy_x / beta))