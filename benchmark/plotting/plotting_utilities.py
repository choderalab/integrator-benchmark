import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os
from benchmark import FIGURE_PATH
figure_directory = FIGURE_PATH
figure_format = ".jpg"

from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy

def generate_figure_filename(filename):
    return os.path.join(FIGURE_PATH, filename)

def savefig(name):
    plt.savefig("{}{}{}".format(figure_directory, name, figure_format), dpi=300)

def plot_results(results, name=""):
    """Given a list of (label, data) results, plot histograms
    on the same axis and save result."""
    style = {"bins": 50,
             "histtype":"stepfilled",
             "normed":True,
             "alpha":0.5
             }

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (label, data) in results:
        ax.hist(data, label=label, **style)

    lgd = ax.legend(loc=(1,0))
    ax.set_xlabel('Kinetic energy ({})'.format(str(ke_unit)))
    ax.set_ylabel('Frequency')
    ax.set_title('WaterBox kinetic energy distribution')
    fig.savefig('kinetic_energy_check_{}.jpg'.format(name), dpi=300,
                bbox_extra_artists = (lgd,), bbox_inches = 'tight')
    plt.close()


def plot(results, name=""):
    """Given a results dictionary that maps scheme strings to work trajectories / DeltaFNeq estimates, plot
    the trajectories and their averages, and print a summary."""
    schemes = results.keys()

    # get min and max
    y_min, y_max = np.inf, -np.inf
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])
        y_min_ = min(np.min(Fs), np.min(Rs))
        y_max_ = max(np.max(Fs), np.max(Rs))
        if y_min_ < y_min: y_min = y_min_
        if y_max_ > y_max: y_max = y_max_

    # plot individuals
    for scheme in schemes:

        # plot the raw shadow work trajectories
        plt.figure()
        plt.title(scheme + get_summary_string(results[scheme], linebreaks=False))

        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        traj_style = {"linewidth": 0.1, "alpha": 0.3}
        for i in range(len(Fs)):
            plt.plot(x_F, Fs[i], color='blue', **traj_style)
            plt.plot(x_R, Rs[i], color='green', **traj_style)

        plt.ylim(y_min, y_max)
        plt.xlabel('# steps')
        plt.ylabel('Shadow work')
        savefig('{}_work_trajectories_{}'.format(name, scheme))

        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color="blue")

        plt.plot(x_R, R_mean, color="green")

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color="blue", alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color="green", alpha=0.3)

        savefig('{}_averaged_work_trajectories_{}'.format(name, scheme))
        plt.close()
    # also make a comparison figure with all of the integrators, just with the confidence bands
    # instead of the full trajectories
    colors = dict(zip(schemes, "blue green orange purple darkviolet".split()))
    plt.figure()
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        # plot the averages
        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color=colors[scheme], label=scheme)
        plt.plot(x_R, R_mean, color=colors[scheme])

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color=colors[scheme], alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color=colors[scheme], alpha=0.3)

        # plot vertical line where midpoint operator is applied
        # plt.vlines((x_F[-1] + x_R[0])/2.0, y_min, y_max, linestyles='--', color='grey')

    plt.xlabel('# steps')
    plt.ylabel('Shadow work')
    plt.title('Comparison')
    plt.legend(fancybox=True, loc='best')
    savefig('{}_averaged_work_trajectories_comparison'.format(name))
    plt.close()


def plot_scheme_comparison(target_filename, name):
    """Plot timestep on the x axis vs. DeltaF_neq on the y axis,
    with a separate curve for each scheme.

    results is a dictionary containing two dictionaries, one for the configuration marginal
    and one for the
        i.e. results[marginal][(name, timestep)] = W_shads_F, W_shads_R
    """
    with open(target_filename, "rb") as f:
        results = pickle.load(f)

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
                    DeltaF_neq, sq_unc = estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R)
                    DeltaF_neqs.append(DeltaF_neq)
                    sq_uncs.append(sq_unc)

                DeltaF_neqs = np.array(DeltaF_neqs)
                sq_uncs = np.array(sq_uncs)
                uncs = 1.96 * np.sqrt(sq_uncs)
                plt.hlines(0, min(timesteps), max(timesteps), linestyles="--")
                plt.errorbar(timesteps, DeltaF_neqs, yerr=uncs, label="{} ({})".format(scheme, marginal))
                # plt.plot(timesteps, DeltaF_neqs, label="{} ({})".format(scheme, marginal))
                # plt.fill_between(timesteps, DeltaF_neqs - uncs, DeltaF_neqs + uncs, alpha=0.3, color='grey')

        plt.legend(loc='best', fancybox=True)
        plt.xlabel("$\Delta t$")
        plt.ylabel("$\Delta F_{neq}$")
        plt.savefig(generate_figure_filename("scheme_comparison_{}.jpg".format(name)), dpi=300)
        plt.close()

    plot_curves(results)
