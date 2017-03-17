import numpy as np
from benchmark import DATA_PATH
import os

from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.evaluation.analysis import compute_free_energy

from estimate_equilibrium_free_energy import x_marginal_free_energy, equilibrium_free_energy
from quartic_simple import potential, kinetic_energy, beta

def parse_condition_from_filename(fname):
    """
    >>>parse_condition_from_filename("quartic_xv_BAOAB_0.25.npy")
    BAOAB, 0.25
    """
    scheme, timestep_ = fname.split("_")[-2:]
    timestep = float(timestep_.split(".npy")[0])
    return scheme, timestep

def estimate_ground_truth_kl_divergences():
    fnames = glob(os.path.join(DATA_PATH, "quartic_xv_*.npy"))

    # get list of timesteps
    timesteps = set()

    okay_fnames = []
    for fname in fnames:
        scheme, timestep = parse_condition_from_filename(fname)

        try:
            xv = np.load(fname)[::1000]
            timesteps.add(timestep)
            okay_fnames.append(fname)
        except:
            print("problem with {}!".format(fname))


    timesteps = sorted(list(timesteps))
    index_of_timestep = dict(zip(timesteps, range(len(timesteps))))

    baoab_conf = np.zeros(len(timesteps))
    baoab_joint = np.zeros(len(timesteps))
    aboba_conf = np.zeros(len(timesteps))
    aboba_joint = np.zeros(len(timesteps))

    for fname in okay_fnames:
        print(fname)
        scheme, timestep = parse_condition_from_filename(fname)

        i = index_of_timestep[timestep]
        
        xv = np.load(fname)[::50]
        print(len(xv))
        F, F_conf = compute_free_energy(xv, potential, kinetic_energy, beta)

        D_KL_joint = beta * (F - equilibrium_free_energy)
        D_KL_conf = beta * (F_conf - x_marginal_free_energy)

        if scheme == "VVVR":
            aboba_conf[i] = D_KL_conf
            aboba_joint[i] = D_KL_joint
        else:
            baoab_conf[i] = D_KL_conf
            baoab_joint[i] = D_KL_joint

    return timesteps, baoab_conf, baoab_joint, aboba_conf, aboba_joint

def plot_kl_divergences(timesteps, baoab_conf, baoab_joint, aboba_conf, aboba_joint):
    plt.figure()
    plt.plot(timesteps, baoab_joint, color="green", label="VRORV (x,v)")
    plt.plot(timesteps, baoab_conf, linestyle="--",color="green", label="VRORV (x)")

    plt.plot(timesteps, aboba_joint, color="blue", label="RVOVR (x,v)")
    plt.plot(timesteps, aboba_conf, linestyle="--", color="blue", label="RVOVR (x)")

    plt.xlabel("Timestep")
    plt.ylabel("$\mathcal{D}_{KL}$")

    plt.legend(loc="best", fancybox=True)
    plt.savefig("ground-truth-kl-divergences.jpg", dpi=300)
    plt.close()

if __name__ == "__main__":
    timesteps, baoab_conf, baoab_joint, aboba_conf, aboba_joint = estimate_ground_truth_kl_divergences()
    plot_kl_divergences(timesteps, baoab_conf, baoab_joint, aboba_conf, aboba_joint)