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

from quartic_simple import beta, velocity_scale, p, log_p, potential, m, sigma2, v_density

data_range = (-2.5, 2.5)
n_bins = 100
bin_edges = np.linspace(data_range[0], data_range[1], num=n_bins)
one_d_hist_args = {"bins": n_bins * 10, "range": data_range, "density": True}

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

def plot_equilibrium_joint_histogram(bin_edges):
    x_eq_hist, v_eq_hist, joint_eq_hist = compute_exact_joint_histogram(bin_edges)

    plt.figure()
    plt.imshow(joint_eq_hist, cmap="Blues")
    plt.title(r"$\pi(x,v)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$", rotation=90)
    plt.savefig('quartic_eq_joint_dist.jpg')
    plt.close()

def plot_equilibrium_joint_histogram_w_x_marginal(bin_edges, loc="above" # or "below"
                                                   ):
    """loc is the location of the marginal plot relative to the image."""
    image_height_factor = 3 # in multiples of the height of the x-marginal plot

    plt.figure()


    if loc == "above":
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, image_height_factor])
        image_ax = plt.subplot(gs[1, 0])
        marginal_ax = plt.subplot(gs[0, 0], sharex=image_ax)
    elif loc == "below":
        gs = gridspec.GridSpec(2, 1, height_ratios=[image_height_factor, 1])
        image_ax = plt.subplot(gs[0, 0])
        marginal_ax = plt.subplot(gs[1, 0], sharex=image_ax)
    else:
        raise(Exception("`loc` must be either 'above' or 'below'"))

    image_ax.imshow(joint_eq_hist, origin='lower', extent=list(data_range)*2,
              aspect='auto', cmap="Blues", alpha=0.3)
    #image_ax.contour(joint_eq_hist, extent=list(data_range)*2, cmap="Blues")
    image_ax.axis('off')

    x_space = (bin_edges[1:] + bin_edges[:-1]) / 2

    marginal_ax.plot(x_space, x_eq_hist, linewidth=2, color="blue", clip_on=False)
    marginal_ax.fill_between(x_space, x_eq_hist, alpha=0.1, color="blue")
    marginal_ax.axis("off")


    xlabel = "$x$"
    if loc == "below":
        marginal_ax.set_xlabel(xlabel)
    else:
        image_ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.savefig(generate_figure_filename('quartic_eq_joint_dist_w_x_marginal.jpg'), dpi=300)
    plt.close()

def get_sampled_joint_histogram(xv):
    joint_sampled_dist, _, _ = np.histogram2d(xv[:, 1], xv[:, 0], bins=bin_edges,
                                                              normed=True)
    return joint_sampled_dist

def normalize(p):
    return p / np.sum(p)

def difference_between_histograms(p, q):
    return p - q

def log_difference_between_histograms(p, q):
    return np.nan_to_num(np.log(p)) - np.nan_to_num(np.log(q))

def KL_integrand_between_histograms(p, q):
    return p * log_difference_between_histograms(p, q)

def process_2d_histogram(histogram):
    Z = np.sum(joint_eq_hist)
    smoothed = histogram
    smoothed = gaussian_filter(histogram, 3.0)
    return Z * smoothed / np.sum(smoothed)
    #return normalize(smoothed)

def process_1d_histogram(histogram):
    smoothed = gaussian_filter1d(histogram, 20.0)
    return normalize(smoothed)

hi_res_bin_edges = np.linspace(data_range[0], data_range[1], one_d_hist_args["bins"] + 1)
hi_res_marginal = compute_exact_histogram(p, hi_res_bin_edges)
pi_x = process_1d_histogram(hi_res_marginal)
x_space = hi_res_bin_edges[1:]

def process_sampled_hist(xv):
    joint_sampled_hist = get_sampled_joint_histogram(xv)
    return process_2d_histogram(joint_sampled_hist)

def process_image(xv):
    joint_eq_hist_ = process_2d_histogram(joint_eq_hist)
    joint_sampled_hist = process_sampled_hist(xv)

    difference = difference_between_histograms(joint_sampled_hist, joint_eq_hist_)
    # KL_integrand = KL_integrand_between_histograms(joint_sampled_hist, joint_eq_hist_)

    image = difference
    #image = KL_integrand

    image = np.nan_to_num(image)

    return image

def plot_image(ax, image, vmin, vmax):
    #ax.imshow(image, origin='lower', extent=list(data_range) * 2, aspect=1, cmap="bwr",
    #          vmin=vmin, vmax=vmax, alpha=0.1)

    ax.contour(image, extent=list(data_range) * 2,
               vmin=vmin, vmax=vmax, cmap="bwr", alpha=1, linewidths=1, aspect=1)

    v_scale = 0.5
    ax.hlines(0, data_range[0], data_range[1], linewidth=0.5)
    ax.vlines(0, data_range[0]*v_scale, data_range[1]*v_scale, linewidth=0.5)
    ax.axis('off')

def plot_marginal_error_curve(ax, curve, ymin, ymax):
    """Plot the curve on the axis, with a horizontal grey line at zero,
    and shading the area between the curve and the x-axis red for positive, blue for negative"""

    #ax.plot(x_space, curve, linewidth=0.5, color="grey", clip_on=False)
    #ax.hlines(0, x_space[0], x_space[-1], color="grey", linestyles="--")
    ax.vlines(0, ymin, ymax, linewidth=0.5)
    ax.plot(x_space, np.maximum(curve, 0), color="red")
    ax.plot(x_space, np.minimum(curve, 0), color="blue")
    ax.fill_between(x_space, np.maximum(curve, 0), color="red", alpha=0.1)
    ax.fill_between(x_space, np.minimum(curve, 0), color="blue", alpha=0.1)
    ax.axis("off")
    ax.set_ylim(ymin, ymax)

def get_x_marginal_density(xv):
    sampled_x_hist, bin_edges = np.histogram(xv[:, 0], **one_d_hist_args)
    rho_x = process_1d_histogram(sampled_x_hist)
    return rho_x

def get_x_marginal_kl_div_from_eq(xv):
    rho_x = get_x_marginal_density(xv)
    return np.mean(KL_integrand_between_histograms(rho_x, pi_x))

def get_joint_kl_div_from_eq(xv):
    hist2d = process_sampled_hist(xv)
    return np.mean(KL_integrand_between_histograms(hist2d, joint_eq_hist))

def get_marginal_error_curve(xv):
    rho_x = get_x_marginal_density(xv)

    # curve = KL_integrand_between_histograms(rho_x, pi_x)
    curve = difference_between_histograms(rho_x, pi_x)
    return curve

def get_max_abs_val(fnames):
    abs_val = 0
    for fname in fnames:
        abs_val = max(abs_val, np.max(np.abs(process_image(load_samples(fname)[0]))))
    return abs_val

def plot_array_of_joint_errors(timesteps, xv_dict_baoab, xv_dict_vvvr):
    """
    timesteps : array

    xv_dict_baoab : maps timestep -> xv array

    xv_dict_vvvr : maps timestep -> xv array
    """
    image_height_factor = 1 # in multiples of the height of the x-marginal plot

    plt.figure()
    n_plots = len(timesteps)

    # each column contains x-marginal, image, image, x-marginal
    gs = gridspec.GridSpec(nrows=4, ncols=n_plots, hspace=0, wspace=0, height_ratios=[1, image_height_factor, 1, image_height_factor])


    data = []
    max_image_abs_val = 0
    max_marginal_abs_val = 0

    update_running_max = lambda current_max, data : max(current_max, np.max(np.abs(data)))

    # load data / generate images
    for i, timestep in enumerate(tqdm(timesteps)):
        xv_baoab = xv_dict_baoab[timestep]
        xv_vvvr = xv_dict_vvvr[timestep]

        baoab_image = process_image(xv_baoab)
        vvvr_image = process_image(xv_vvvr)
        max_image_abs_val = update_running_max(max_image_abs_val, baoab_image)
        max_image_abs_val = update_running_max(max_image_abs_val, vvvr_image)

        baoab_marginal = get_marginal_error_curve(xv_baoab)
        vvvr_marginal = get_marginal_error_curve(xv_vvvr)
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

        # plot errors in joint distribution
        plot_image(upper_image_ax, baoab_image, vmin, vmax)
        plot_image(lower_image_ax, vvvr_image, vmin, vmax)

        # plot x-marginal errors
        plot_marginal_error_curve(upper_marginal_ax, baoab_marginal, ymin, ymax)
        plot_marginal_error_curve(lower_marginal_ax, vvvr_marginal, ymin, ymax)

    plt.tight_layout(pad=1.0)
    plt.savefig(generate_figure_filename('quartic_eq_joint_dist_array_w_x_marginals.pdf'))
    plt.close()

    return timesteps, data

def load_samples(fname):
    """Load the (x,v) samples from fname, and fetch the condition_name"""
    xv = np.load(fname)
    print(xv.shape)
    condition_name = fname[fname.find("quartic_xv") + 10:-4]
    return xv, condition_name

def load_dictionaries(fnames):
    """Loads the samples into two dictionaries."""
    xv_dict_baoab = {}
    xv_dict_vvvr = {}

    for fname in sorted(fnames):
        xv, condition_name = load_samples(fname)

        # between the last underscore and .jpg,
        timestep_string = condition_name.split("_")[-1]
        timestep = float(timestep_string)

        if "VVVR" in condition_name:
            xv_dict_vvvr[timestep] = xv
        else:
            xv_dict_baoab[timestep] = xv

    return xv_dict_baoab, xv_dict_vvvr


def plot_kl_divergences(timesteps, baoab_conf, baoab_joint, vvvr_conf, vvvr_joint):
    plt.figure()
    plt.plot(timesteps, baoab_joint, color="green", label="VRORV (x,v)")
    plt.plot(timesteps, baoab_conf, linestyle="--",color="green", label="VRORV (x)")

    plt.plot(timesteps, vvvr_joint, color="blue", label="RVOVR (x,v)")
    plt.plot(timesteps, vvvr_conf, linestyle="--", color="blue", label="RVOVR (x)")

    plt.xlabel("Timestep")
    plt.ylabel("$\mathcal{D}_{KL}$")

    plt.legend(loc="best", fancybox=True)
    plt.savefig("kl-divergences-from-histograms.jpg", dpi=600)
    plt.close()

if __name__ == "__main__":
    plot_equilibrium_joint_histogram_w_x_marginal(bin_edges)

    fnames = glob(os.path.join(DATA_PATH, "quartic_xv_*.npy"))

    print("loading dictionaries...")
    xv_dict_baoab, xv_dict_vvvr = load_dictionaries(fnames)
    print("plotting...")
    timesteps, data = plot_array_of_joint_errors([0.6, 0.8, 1.0, 1.2], xv_dict_baoab, xv_dict_vvvr)
    #plot_kl_divergences(timesteps, *get_kl_divergences(data))
