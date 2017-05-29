import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from pickle import load

from benchmark import DATA_PATH, FIGURE_PATH
from benchmark.evaluation import estimate_nonequilibrium_free_energy

fnames = glob(os.path.join(DATA_PATH, "*.pkl"))

figure_1_fnames = []
for fname in fnames:
    if "1_splittings" in fname:
        figure_1_fnames.append(fname)

scheme_names = []
timesteps = []
for fname in figure_1_fnames:
    try:
        with open(fname, "rb") as f:
            x = load(f)
            if x["descriptor"]["marginal"] == "full" and x["descriptor"]["collision_rate_name"] == "low":
                scheme_names.append(x["descriptor"]["splitting_name"])
                timesteps.append(x["descriptor"]["timestep_in_fs"])
    except:
        pass
timesteps = np.array(sorted(list(set(timesteps))))


def get_n_dof(system_name):
    if "alanine".upper() in system_name.upper():
        # return 22 * 3
        return 54  # with constraints
    elif "dhfr".upper() in system_name.upper():
        # return 23558 * 3
        return 48384  # with constraints
    elif "t4".upper() in system_name.upper():
        # return 2621 * 3
        return 6540  # with constraints
    elif "water".upper() in system_name.upper():
        # return 1503 * 3
        return 3006  # with constraints
    else:
        raise (Exception("System name not recognized"))


def get_curves(fnames, timesteps, scheme_names, system_name,
               marginal="full", collision_rate_name="low", ):
    curves = {}  # map from scheme_name to DeltaF_neq curve
    error_curves = {}  # map from scheme_name to error curve
    for scheme in set(scheme_names):
        curves[scheme] = np.zeros(len(timesteps)) * np.nan
        error_curves[scheme] = np.zeros(len(timesteps)) * np.nan

    t_to_index = dict(zip(timesteps, range(len(timesteps))))

    for fname in fnames:
        try:
            with open(fname, "rb") as f:
                x = load(f)
                if x["descriptor"]["splitting_name"] in scheme_names and x["descriptor"]["marginal"] == marginal and \
                                x["descriptor"]["collision_rate_name"] == collision_rate_name and x["descriptor"][
                    "system_name"] == system_name:
                    scheme, t = x["descriptor"]["splitting_name"], x["descriptor"]["timestep_in_fs"]
                    DeltaF_neq, err = estimate_nonequilibrium_free_energy(*x["result"])
                    i = t_to_index[t]

                    n_dof = get_n_dof(system_name)
                    curves[scheme][i] = DeltaF_neq / n_dof
                    error_curves[scheme][i] = np.sqrt(err) / n_dof
        except:
            pass

    return curves, error_curves


def plot_curves(timesteps, curves, error_curves):
    for scheme in sorted(curves.keys()):
        if np.isnan(curves[scheme]).sum() == len(curves[scheme]):
            print("Note: {} is all NaNs...".format(scheme))

        y, yerr = curves[scheme], error_curves[scheme]
        plt.errorbar(timesteps, y, yerr=yerr, label=scheme)

    plt.hlines(0, 0, max(timesteps), linestyles='--')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$\Delta F_{neq}$')


# Plot figure 1
plt.figure(figsize=(6, 4))
ax1 = plt.subplot(1, 2, 1)
plot_curves(timesteps, *get_curves(figure_1_fnames, timesteps, sorted(list(set(scheme_names))), "DHFR (constrained)",
                                   marginal="full"))
plt.title("Phase-space error")
plt.legend()
plt.subplot(1, 2, 2, sharey=ax1)
plot_curves(timesteps, *get_curves(figure_1_fnames, timesteps, sorted(list(set(scheme_names))), "DHFR (constrained)",
                                   marginal="configuration"))
plt.title("Configuration-space error")
plt.tight_layout(rect=(0, 0, 2, 1))

plt.savefig(os.path.join(FIGURE_PATH, "figure_1_draft.pdf"), bbox_inches="tight")
plt.close()

# Plot systems comparison figure
figure_2_fnames = []
for fname in fnames:
    if "2_systems" in fname:
        figure_2_fnames.append(fname)

system_names = []
for fname in figure_2_fnames:
    with open(fname, "rb") as f:
        system_names.append(load(f)["descriptor"]["system_name"])
system_names = list(set(system_names))

marginals = ["full", "configuration"]

nrows = len(system_names)
ncols = len(marginals)
plt.figure(figsize=(8, 12))
plot_number = 1
for i in range(nrows):
    for j in range(ncols):
        plt.subplot(nrows, ncols, plot_number)

        plot_curves(timesteps,
                    *get_curves(figure_2_fnames, timesteps, scheme_names, system_names[i], marginal=marginals[j]))

        plot_number += 1

        if i == 0 and j == 0:
            plt.title("Phase-space error")
            plt.legend()
        elif i == 0 and j == 1:
            plt.title("Configuration-space error")

        if i != nrows - 1:
            # except at bottom of figure, remove x-axis labels
            plt.xlabel("")
            # locs, labels = plt.xticks()
            # plt.xticks(locs, [""]*len(locs))

        if j != 0:
            # except at left of figure, remove y-axis labels
            plt.ylabel("")

plt.savefig(os.path.join(FIGURE_PATH, "systems_draft.pdf"))

# Plot geodesic figure
figure_3_fnames = []
for fname in fnames:
    if "3_geodesic" in fname:
        figure_3_fnames.append(fname)

with open(figure_3_fnames[0], "rb") as f:
    system_name = load(f)["descriptor"]["system_name"]

g_timesteps = []
for fname in figure_3_fnames:
    with open(fname, "rb") as f:
        g_timesteps.append(load(f)["descriptor"]["timestep_in_fs"])

g_timesteps = np.array(sorted(list(set(g_timesteps))))

g_scheme_names = []
for fname in figure_3_fnames:
    with open(fname, "rb") as f:
        g_scheme_names.append(load(f)["descriptor"]["splitting_name"])
g_scheme_names = sorted(list(set(g_scheme_names)))

base_schemes = set([s.split()[0] for s in g_scheme_names])

# couple the y-axes of all plots to the first one...
is_first = True

for base_scheme in base_schemes:
    g_schemes = sorted([s for s in g_scheme_names if base_scheme in s])
    print(g_schemes)
    plt.figure(figsize=(6, 4))

    if is_first:
        ax1 = plt.subplot(1, 2, 1)
        is_first == False
    else:
        plt.subplot(1, 2, 1, sharey=ax1)

    g_results = get_curves(figure_3_fnames, g_timesteps, g_schemes, system_name, marginal="full")
    plot_curves(g_timesteps, *g_results)
    plt.legend()
    plt.title("Phase-space error")

    plt.subplot(1, 2, 2, sharey=ax1)
    g_results = get_curves(figure_3_fnames, g_timesteps, g_schemes, system_name, marginal="configuration")
    plot_curves(g_timesteps, *g_results)
    plt.title("Configuration-space error")

    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_PATH, "geodesic_draft_{}.pdf".format(base_scheme)))
