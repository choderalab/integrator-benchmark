# Get precise estimates for the equilibrium free energy, by
# numerical quadrature.
# We will use this for:
# * Validating the nonparametric estimator of entropy and free energy
# * Computing nonequilibrium free energy differences

#from quartic_simple import potential, kinetic_energy, log_p, log_v_density, beta
import numpy as np
from benchmark.testsystems import quartic

x = np.linspace(-10, 10, 10000)

Z = np.trapz(np.exp(quartic.log_q(x)), x)

def log_p(x):
    return quartic.log_q(x) - np.log(Z)

def kinetic_energy(v):
    return 0.5 * quartic.mass * v**2

beta = quartic.beta
sigma2 = quartic.velocity_scale**2
potential = quartic.potential

def log_v_density(v):
    return -v ** 2 / (2 * sigma2) - np.log((np.sqrt(2 * np.pi * sigma2)))

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

equilibrium_free_energy = average_energy - entropy / beta
x_marginal_free_energy = average_potential - entropy_x / beta

print("Equilibrium free energy: {}".format(equilibrium_free_energy))
print("x-marginal equilibrium free energy: {}".format(x_marginal_free_energy))

# For comparison, let's estimate the entropy using the k-nearest-neighbor estimator
if __name__ == "__main__":
    from benchmark.evaluation.entropy import estimate_entropy, estimate_marginal_entropies

    from benchmark.testsystems import quartic
    n_samples = 50000
    _ = quartic.sample_x_from_equilibrium()
    eq_xs = quartic.x_samples
    np.random.shuffle(eq_xs)
    eq_xs = eq_xs[:n_samples]

    eq_vs = np.random.randn(n_samples) * quartic.velocity_scale

    xv = np.vstack((eq_xs, eq_vs)).T
    print(xv.shape)

    for k in range(1, 10):
        print("k={}".format(k))
        estimated_marginal_entropies = estimate_marginal_entropies(xv, k)
        estimated_entropy_x, estimated_entropy_v = estimated_marginal_entropies
        print("\tEstimated marginal entropies of p(x) and p(v): {}".format(estimated_marginal_entropies))
        estimated_joint_entropy = estimate_entropy(xv, k)
        print("\tEstimated joint entropy of p(x,v): {}".format(estimated_joint_entropy))

        estimated_equilibrium_free_energy = average_energy - estimated_joint_entropy / beta
        estimated_x_marginal_free_energy = average_potential - estimated_entropy_x / beta
        print("\tEstimated equilibrium free energy: {}".format(estimated_equilibrium_free_energy))
        print("\tEstimated x-marginal equilibrium free energy: {}".format(estimated_x_marginal_free_energy))