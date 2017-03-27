import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from benchmark import DATA_PATH
from benchmark.plotting import generate_figure_filename
import os
from glob import glob
from tqdm import tqdm

from benchmark.experiments.toy.axis_aligned_joint_plot import joint_eq_hist, plot_marginal_error_curve, difference_between_histograms, pi_x, plot_image, data_range

from benchmark.experiments.toy.axis_aligned_joint_plot import process_1d_histogram, process_2d_histogram
def normalize(x):
    return x / np.sum(x)


#pi_x = process_1d_histogram(normalize(pi_x))

def process_image(joint_sampled_hist):

    joint_eq_hist_ = process_2d_histogram(joint_eq_hist)

    joint_sampled_hist = process_2d_histogram(joint_sampled_hist.T)

    difference = difference_between_histograms(joint_sampled_hist, joint_eq_hist_)

    image = difference

    image = np.nan_to_num(image)

    return image

def get_marginal_error_curve(sampled_x_hist):
    rho_x = process_1d_histogram(sampled_x_hist)
    curve = difference_between_histograms(rho_x, pi_x)
    return curve

def plot_array_of_joint_errors(baoab_joint_hists, baoab_x_hists,
                               vvvr_joint_hists, vvvr_x_hists):
    """
    timesteps : array

    xv_dict_baoab : maps timestep -> xv array

    """
    image_height_factor = 5 # in multiples of the height of the x-marginal plot

    plt.figure()
    n_plots = len(baoab_joint_hists)

    # each column contains x-marginal, image, image, x-marginal
    gs = gridspec.GridSpec(nrows=4, ncols=n_plots, height_ratios=[1, image_height_factor, 1, image_height_factor])


    data = []
    max_image_abs_val = 0
    max_marginal_abs_val = 0

    update_running_max = lambda current_max, data : max(current_max, np.max(np.abs(data)))

    # load data / generate images
    for i in range(n_plots):

        baoab_image = process_image(baoab_joint_hists[i])
        vvvr_image = process_image(vvvr_joint_hists[i])

        max_image_abs_val = update_running_max(max_image_abs_val, baoab_image)
        max_image_abs_val = update_running_max(max_image_abs_val, vvvr_image)

        baoab_marginal = get_marginal_error_curve(baoab_x_hists[i])
        vvvr_marginal = get_marginal_error_curve(vvvr_x_hists[i])

        max_marginal_abs_val = update_running_max(max_marginal_abs_val, baoab_marginal)
        max_marginal_abs_val = update_running_max(max_marginal_abs_val, vvvr_marginal)

        data.append((baoab_image, vvvr_image, baoab_marginal, vvvr_marginal))

    # get the scales to use
    vmin, vmax = - max_image_abs_val, max_image_abs_val
    ymin, ymax = - max_marginal_abs_val, max_marginal_abs_val

    # generate plots
    for i, (baoab_image, vvvr_image, baoab_marginal, vvvr_marginal) in enumerate(tqdm(data)):
        upper_image_ax = plt.subplot(gs[1, i])
        upper_marginal_ax = plt.subplot(gs[0, i], sharex=upper_image_ax)


        lower_image_ax = plt.subplot(gs[3, i], sharex=upper_image_ax)
        lower_marginal_ax = plt.subplot(gs[2, i], sharex=upper_image_ax)

        #for ax in [upper_image_ax, lower_image_ax]:
        #    ax.set_aspect(2)

        # plot errors in joint distribution
        plot_image(upper_image_ax, baoab_image, vmin, vmax)
        plot_image(lower_image_ax, vvvr_image, vmin, vmax)

        # plot x-marginal errors
        plot_marginal_error_curve(upper_marginal_ax, baoab_marginal, ymin, ymax)
        plot_marginal_error_curve(lower_marginal_ax, vvvr_marginal, ymin, ymax)

    plt.tight_layout(pad=0.0)
    plt.savefig(generate_figure_filename('quartic_eq_joint_dist_array_w_x_marginals_cycle.jpg'), dpi=300)
    plt.close()

def plot_all():
    baoab_joint_hists, baoab_x_hists, vvvr_joint_hists, vvvr_x_hists = [], [], [], []
    for i in range(6):
        vvvr_joint_hists.append(np.load("vvvr_joint_hist_{}.npy".format(i)))

        baoab_joint_hists.append(np.load("baoab_joint_hist_{}.npy".format(i)))

        vvvr_x_hists.append(np.load("vvvr_x_hist_{}.npy".format(i)))

        baoab_x_hists.append(np.load("baoab_x_hist_{}.npy".format(i)))

    plot_array_of_joint_errors(baoab_joint_hists=baoab_joint_hists,
                               baoab_x_hists=baoab_x_hists,
                               vvvr_joint_hists=vvvr_joint_hists,
                               vvvr_x_hists=vvvr_x_hists)

if __name__ == "__main__":
    plot_all()
