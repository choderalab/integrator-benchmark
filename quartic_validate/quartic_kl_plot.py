# Compare the results from numerical ground truth (quartic_kl_histograms.py)
# and the noneq estimates (quartic_kl_validation.py)

from glob import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark.testsystems import quartic

from pickle import load
import os
from benchmark.evaluation import estimate_nonequilibrium_free_energy

import seaborn.apionly as sns
palette = sns.color_palette(n_colors=4)
scheme_names = ["OVRVO", "ORVRO", "RVOVR", "VRORV"]
colors = dict(zip(scheme_names, palette))

fnames = glob("quartic_results/0*.pkl")
print(len(fnames))

np.random.shuffle(fnames)

print("Loading dataset...")
timesteps = []
schemes = []
for fname in tqdm(fnames):
    with open(fname, "rb") as f:
        result = load(f)
        schemes.append(result['descriptor']['splitting'])
        timesteps.append(result['descriptor']['timestep'])
timesteps = sorted(list(set(timesteps)))
timestep_to_i = dict(zip(timesteps, range(len(timesteps))))
schemes = sorted(list(set(schemes)))

def get_results():
    results_conf = {}
    results_full = {}

    for fname in tqdm(fnames):
        with open(fname, "rb") as f:
            result = load(f)
            splitting, timestep = result['descriptor']['splitting'], result['descriptor']['timestep']

            W_F, W_R = result['result']["W_shads_F"], result['result']["W_shads_R"]
            DF, dDF2 = estimate_nonequilibrium_free_energy(W_F, W_R)

            if result['descriptor']['marginal'] == "configuration":
                destination = results_conf
            else:
                destination = results_full

            if (splitting not in destination):
                destination[splitting] = (np.nan * np.zeros(len(timesteps)), np.nan * np.zeros(len(timesteps)))
            destination[splitting][0][timestep_to_i[timestep]] = DF
            destination[splitting][1][timestep_to_i[timestep]] = np.sqrt(dDF2)
    return results_conf, results_full

results_conf, results_full = get_results()

# load the histograms and turn them into KL divergence arrays

from pickle import dump
with open("quartic_results/conf_results.pkl", "wb") as f:
    dump(results_conf, f)

with open("quartic_results/full_results.pkl", "wb") as f:
    dump(results_full, f)

# load numerical results
conf_fname, full_fname = 'quartic_results/conf_results.pkl', 'quartic_results/full_results.pkl'
with open(conf_fname, "rb") as f:
    conf_results = load(f)
with open(full_fname, "rb") as f:
    full_results = load(f)

# load noneq results
fnames = glob('quartic_results/0*.npz')

timesteps = []
for fname in fnames:
    file = np.load(fname)
    dt, scheme = file['description']
    timesteps.append(float(dt))
timesteps = sorted(list(set(timesteps)))
timestep_to_i = dict(zip(timesteps, range(len(timesteps))))

colormaps = {
    "RVOVR": "Greens",
    "ORVRO": "Oranges",
    "OVRVO": "Blues",
    "VRORV": "Reds"
}


x = np.linspace(-10, 10, 10000)

Z = np.trapz(np.exp(quartic.log_q(x)), x)

def log_p(x):
    return quartic.log_q(x) - np.log(Z)

def kinetic_energy(v):
    return 0.5 * quartic.mass * v**2

mass = quartic.mass
beta = quartic.beta
velocity_scale = np.sqrt(1.0 / (beta * mass))
sigma2 = quartic.velocity_scale**2
potential = quartic.potential

def log_v_density(v):
    return -v ** 2 / (2 * sigma2) - np.log((np.sqrt(2 * np.pi * sigma2)))

def p(x):
    return np.exp(log_p(x))

def v_density(v):
    return np.exp(log_v_density(v))

data_range = (-2.5, 2.5)
n_bins = 50
bin_edges = np.linspace(data_range[0], data_range[1], num=n_bins)

one_d_hist_args = {"bins": n_bins * 2, "range": data_range, "density": True}
one_d_bin_edges = np.linspace(data_range[0], data_range[1], num=n_bins*2 + 1)


def normalize_histogram(hist, bin_edges):
    x_range = bin_edges[-1] - bin_edges[0]
    sum_y = np.sum(hist)
    Z = (sum_y / x_range)
    return hist / Z

def compute_exact_histogram(density, bin_edges):
    exact_hist = np.zeros(len(bin_edges) - 1)
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        x_ = np.linspace(left, right, 1000)
        y_ = density(x_)
        exact_hist[i] = np.trapz(y_, x_)
    # let's double-check to make sure this histogram is normalized
    return normalize_histogram(exact_hist, bin_edges)

def compute_exact_joint_histogram(bin_edges):
    x_eq_hist = compute_exact_histogram(p, bin_edges)
    v_eq_hist = compute_exact_histogram(v_density, bin_edges)
    joint_eq_hist = np.outer(v_eq_hist, x_eq_hist)
    return x_eq_hist, v_eq_hist, joint_eq_hist

x_eq_hist, v_eq_hist, joint_eq_hist = compute_exact_joint_histogram(bin_edges)
x_eq_hist = compute_exact_histogram(p, one_d_bin_edges)

from scipy.stats import entropy


def get_entropies(scheme="OVRVO"):
    entropies = []
    for dt in timesteps:
        entropies.append(entropy(one_d_hists[(dt, scheme)], x_eq_hist))
    return np.array(entropies)


one_d_hists = {}

for fname in fnames:
    file = np.load(fname)
    dt, scheme = file['description']
    dt = float(dt)

    c = 1.0 * file['counts_1d']
    c /= np.sum(c)
    c *= 5
    Z = c.T  # - x_eq_hist

    one_d_hists[(dt, scheme)] = Z


def get_2d_entropies(scheme="OVRVO"):
    eps = 0.000
    entropies = []
    for dt in timesteps:
        entropies.append(entropy(two_d_hists[(dt, scheme)].flatten() + eps, joint_eq_hist.flatten() + eps))
    return np.array(entropies)


two_d_hists = {}
two_d_diffs = {}

for fname in fnames:
    file = np.load(fname)
    dt, scheme = file['description']
    dt = float(dt)

    c = file['counts_2d']
    c /= np.sum(c)
    c *= 25
    Z = c.T  # - joint_eq_hist

    two_d_hists[(dt, scheme)] = Z

    two_d_diffs[(dt, scheme)] = Z - joint_eq_hist

plt.figure(figsize=(8, 4))
ax = plt.subplot(121)

for scheme in colors.keys():
    plt.plot([0] + list(timesteps), [0] + list(get_2d_entropies(scheme)), label=scheme, c=colors[scheme])

    y_mean, y_stdev = full_results[scheme]

    lb = y_mean - 1.96 * y_stdev
    ub = y_mean + 1.96 * y_stdev

    plt.fill_between([0] + list(timesteps), [0] + list(lb), [0] + list(ub), alpha=0.3, color=colors[scheme])

plt.legend(title='schemes')
plt.hlines(0, 0, max(timesteps), linestyles='dotted')
plt.xlim(0, )

plt.xlabel("$\Delta t$")
plt.ylabel(r"$\mathcal{D}_{KL}(\rho \| \pi)$")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.title("Noneq vs numerical KL divergence\nfull (x,v) distribution")

ax = plt.subplot(122, sharey=ax)

plt.title("Noneq vs numerical KL divergence\nconfiguration distribution")

for scheme in colors.keys():
    plt.plot([0] + list(timesteps), [0] + list(get_entropies(scheme)), label=scheme, c=colors[scheme])

    y_mean, y_stdev = conf_results[scheme]

    lb = y_mean - 1.96 * y_stdev
    ub = y_mean + 1.96 * y_stdev

    plt.fill_between([0] + list(timesteps), [0] + list(lb), [0] + list(ub), alpha=0.3, color=colors[scheme])

plt.hlines(0, 0, max(timesteps), linestyles='dotted')
plt.legend(title='schemes', loc="upper left")

plt.xlabel("$\Delta t$")
plt.ylabel(r"$\mathcal{D}_{KL}(\rho_\mathbf{x} \| \pi_\mathbf{x})$")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig('numerical_vs_noneq_KL_divergence.pdf', bbox_inches="tight")
