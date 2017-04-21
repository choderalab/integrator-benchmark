# Here, just generate the plot with timestep on the x axis, histogram-based KL divergence on the y-axis
# Curves for {VVVR, BAOAB} x {configuration, full}

from axis_aligned_joint_plot import joint_eq_hist, load_dictionaries,\
    get_joint_kl_div_from_eq, get_x_marginal_kl_div_from_eq
from glob import glob
import os
from benchmark import DATA_PATH, FIGURE_PATH
from tqdm import tqdm
import numpy as np

fnames = glob(os.path.join(DATA_PATH, "quartic_xv_*.npy"))

xv_dict_baoab, xv_dict_vvvr = load_dictionaries (fnames)
timesteps = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

joint_KL_baoab = []
joint_KL_vvvr = []

marginal_KL_baoab = []
marginal_KL_vvvr = []

for i, timestep in enumerate(tqdm(timesteps)):
    xv_baoab = xv_dict_baoab[timestep]
    xv_vvvr = xv_dict_vvvr[timestep]

    joint_KL_baoab.append(get_joint_kl_div_from_eq(xv_baoab))
    joint_KL_vvvr.append(get_joint_kl_div_from_eq(xv_vvvr))

    marginal_KL_baoab.append(get_x_marginal_kl_div_from_eq(xv_baoab))
    marginal_KL_vvvr.append(get_x_marginal_kl_div_from_eq(xv_vvvr))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"OVRVO": "blue", "VRORV": "green"}
line_style = {"linewidth":2}

plt.figure()
plt.plot(timesteps, joint_KL_baoab, label="VRORV (full)", color=colors["VRORV"], **line_style)
plt.plot(timesteps, joint_KL_vvvr, label="OVRVO (full)", color=colors["OVRVO"], **line_style)
plt.plot(timesteps, marginal_KL_baoab, label="VRORV (conf)", color=colors["VRORV"], linestyle='--', **line_style)
plt.plot(timesteps, marginal_KL_vvvr, label="OVRVO (conf)", color=colors["OVRVO"], linestyle='--', **line_style)

# maybe I should also shade under each
#fill_style = {"alpha":0.1}
#plt.fill_between(timesteps, joint_KL_baoab, color=colors["VRORV"], **fill_style)
#plt.fill_between(timesteps, joint_KL_vvvr, color=colors["OVRVO"], **fill_style)
#plt.fill_between(timesteps, marginal_KL_baoab, color=colors["VRORV"], **fill_style)
#plt.fill_between(timesteps, marginal_KL_vvvr, color=colors["OVRVO"], **fill_style)


plt.xlabel(r'$\Delta t$')
plt.ylabel(r"$D_{KL}$")
#plt.yscale('log')j
plt.legend(loc="best", fancybox=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH, "ground_truth_KL_divs.pdf"))
plt.close()

#plt.figure()
#plt.imshow(KL_integrand_between_histograms(process_sampled_hist(xv_vvvr), joint_eq_hist), cmap='Blues')
#plt.colorbar()
#plt.savefig(os.path.join(FIGURE_PATH, "KL_integrand.jpg"))
#plt.close()