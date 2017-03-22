import numpy as np
from pickle import load
from entropy import estimate_marginal_entropies, estimate_entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R, N_eff=None):
    """Estimate nonequilibrium free energy (and uncertainty) from work measurements.

    Reference: eqs. A1, A2 in http://threeplusone.com/Sivak2013a.pdf
    """
    if N_eff == None: N_eff = len(W_shads_F)
    DeltaF_neq = 0.5 * (np.mean(W_shads_F) - np.mean(W_shads_R))
    # squared uncertainty = [var(W_0->M) + var(W_M->2M) - 2 cov(W_0->M, W_M->2M)] / (4 N_eff)
    sq_uncertainty = (np.var(W_shads_F) + np.var(W_shads_R) - 2 * np.cov(W_shads_F, W_shads_R)[0,1]) / (4 * N_eff)
    return DeltaF_neq, sq_uncertainty

def compute_free_energy(xv, potential, kinetic_energy, beta):
    """Given a potential energy function, kinetic energy function, and inverse-temperature beta,
    compute the free energy as
        F = <E> - S / beta
    and the configuration-marginal free energy as
        F_conf = <U> - S_conf / beta
    """
    x_samples, v_samples = xv[:,0], xv[:,1]
    # print average potential energy
    avg_potential = np.mean(potential(x_samples))

    avg_ke = np.mean(kinetic_energy(v_samples))

    avg_energy = np.mean(potential(x_samples) + kinetic_energy(v_samples))


    entropy_x, entropy_v = estimate_marginal_entropies(xv)
    entropy = estimate_entropy(xv)


    F = avg_energy - entropy / beta
    F_conf = avg_potential - entropy_x / beta
    print("\tF = {:.5f}".format(F))
    print("\tF_conf = {:.5f}".format(F_conf))

    print("\t\t<U> = {:.5f}".format(avg_potential))
    print("\t\t<KE> = {:.5f}".format(avg_ke))
    print("\t\t<E> = {:.5f}".format(avg_energy))
    print("\t\tS_configurations = {:.5f}".format(entropy_x))
    print("\t\tS_momenta = {:.5f}".format(entropy_v))
    print("\t\tS = {:.5f}".format(entropy))

    return F, F_conf


# also estimate configurational temperature

# can also add bootstrapped confidence interval, if we use a different estimator

if __name__ == "__main__":
    print("reading and re-analyzing results")
    name = "alanine_unconstrained_null_results.pkl"
    with open(name, "r") as f: results = load(f)
    for scheme in results.keys():
        print(scheme)
        W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoints = results[scheme]

        histstyle = {"alpha":0.3,
                     "histtype":"stepfilled",
                     "bins":50
                     }

        # plot W_shads_F and W_shads_R distributions
        plt.figure()
        plt.title("W_shads ({})".format(scheme))
        plt.hist(np.array(W_shads_F)[:,-1], label='W_shads_F', **histstyle);
        plt.hist(np.array(W_shads_R)[:,-1], label="W_shads_R", **histstyle);
        plt.legend(loc="best", fancybox=True)
        plt.savefig("W_shad_distributions_{}.jpg".format(scheme), dpi=300)
        plt.close()
