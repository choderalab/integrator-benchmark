import numpy as np
from pickle import load
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



# also estimate configurational temperature

# also add KL divergence tools

# rephrase nonequilibrium free energy in terms of KL divergence

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
