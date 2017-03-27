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

def plot_array_of_joint_errors(scheme, joint_hists, x_hists):
    """
    timesteps : array

    """
    image_height_factor = 5 # in multiples of the height of the x-marginal plot

    plt.figure()
    n_plots = len(joint_hists)

    # each column contains x-marginal, image, image, x-marginal
    gs = gridspec.GridSpec(nrows=2, ncols=n_plots, height_ratios=[1, image_height_factor])


    data = []
    max_image_abs_val = 0
    max_marginal_abs_val = 0

    update_running_max = lambda current_max, data : max(current_max, np.max(np.abs(data)))

    # load data / generate images
    for i in range(n_plots):

        image = process_image(joint_hists[i])

        max_image_abs_val = update_running_max(max_image_abs_val, image)

        marginal = get_marginal_error_curve(x_hists[i])

        max_marginal_abs_val = update_running_max(max_marginal_abs_val, marginal)

        data.append((image, marginal))

    # get the scales to use
    vmin, vmax = - max_image_abs_val, max_image_abs_val
    ymin, ymax = - max_marginal_abs_val, max_marginal_abs_val

    # generate plots
    for i, (image, marginal) in enumerate(tqdm(data)):
        image_ax = plt.subplot(gs[1, i])
        marginal_ax = plt.subplot(gs[0, i], sharex=image_ax)

        # plot errors in joint distribution
        plot_image(image_ax, image, vmin, vmax)

        # plot x-marginal errors
        plot_marginal_error_curve(marginal_ax, marginal, ymin, ymax)

    #plt.tight_layout(pad=0.0)
    plt.savefig(generate_figure_filename('quartic_eq_joint_dist_array_w_x_marginals_cycle_{}.jpg'.format(scheme)), dpi=300)
    plt.close()

def plot_all(schemes):

    for scheme in schemes:
        joint_hists, x_hists = [], []
        for i in range(6):
            joint_hists.append(np.load("{}_joint_hist_{}.npy".format(scheme, i)))
            x_hists.append(np.load("{}_x_hist_{}.npy".format(scheme, i)))


        plot_array_of_joint_errors(scheme, joint_hists=joint_hists, x_hists=x_hists)

if __name__ == "__main__":
    plot_all(schemes=["vvvr", "baoab", "aboba"])
