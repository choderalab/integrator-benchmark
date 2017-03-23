# Analyze and plot the results from `baoab_vs_aboba.py`

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark import DATA_PATH
import os
import numpy as np
import pickle
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.plotting import generate_figure_filename

def plot_results(target_filename, name):
    with open(target_filename, "r") as f:
        results = pickle.load(f)

    # results is a dictionary containing two dictionaries
    # results[marginal][(name, timestep)] = W_shads_F, W_shads_R
    # I want to plot timestep vs. DeltaF_neq for each (name)

    def get_schemes(result_dict):
        schemes = sorted(list(set([key[0] for key in result_dict.keys()])))
        return schemes

    def get_timesteps(result_dict):
        timesteps = sorted(list(set([key[1] for key in result_dict.keys()])))
        return timesteps

    def plot_curves(results):
        plt.figure()
        for marginal in results.keys():
            schemes = get_schemes(results[marginal])
            timesteps = get_timesteps(results[marginal])


            for scheme in schemes:
                DeltaF_neqs = []
                sq_uncs = []

                for timestep in timesteps:
                    W_shads_F, W_shads_R = results[marginal][(scheme, timestep)]
                    W_shads_F = np.array(W_shads_F)
                    W_shads_R = np.array(W_shads_R)
                    DeltaF_neq, sq_unc = estimate_nonequilibrium_free_energy(W_shads_F[:,-1], W_shads_R[:,-1])
                    DeltaF_neqs.append(DeltaF_neq)
                    sq_uncs.append(sq_unc)

                DeltaF_neqs = np.array(DeltaF_neqs)
                sq_uncs = np.array(sq_uncs)
                uncs = 1.96 * np.sqrt(sq_uncs)

                plt.errorbar(timesteps, DeltaF_neqs, yerr=uncs, label="{} ({})".format(scheme, marginal))
                #plt.plot(timesteps, DeltaF_neqs, label="{} ({})".format(scheme, marginal))
                #plt.fill_between(timesteps, DeltaF_neqs - uncs, DeltaF_neqs + uncs, alpha=0.3, color='grey')

        plt.legend(loc='best', fancybox=True)
        plt.savefig(generate_figure_filename("baoab_vs_aboba_{}.jpg".format(name)))
        plt.close()

    plot_curves(results)

if __name__ == "__main__":
    name = "waterbox_constrained"
    target_filename = os.path.join(DATA_PATH, "baoab_vs_aboba_{}.pkl".format(name))
    plot_results(target_filename, name)