from tqdm import tqdm
import numpy as np
import matplotlib
from benchmark.experiments.toy.axis_aligned_joint_plot import bin_edges, one_d_hist_args
# one_d_hist_args["density"] = False

from benchmark.integrators.numba_integrators import vvvr_factory, baoab_factory
from generate_integrator_cycle_plots import plot_all

matplotlib.use('agg')


# define system
np.random.seed(0)

figure_directory = "figures/"  # relative to script
figure_format = ".pdf"

# Define system
beta = 1.0  # inverse temperature
dim = 1  # system dimension

def potential(x): return x**4

def force(x): return - 4.0 * x**3

def reduced_potential(x): return potential(x) * beta

def log_q(x): return - reduced_potential(x)

def q(x): return np.exp(log_q(x))


# normalized density
x = np.linspace(-3, 3, 1000)
x_ = np.linspace(-10,10,10000)
Z = np.trapz(q(x_), x_)
log_Z = np.log(Z)

def p(x): return q(x) / Z
def log_p(x): return log_q(x) - log_Z


m = 10.0  # mass
velocity_scale = np.sqrt(1.0 / (beta * m))
sigma2 = velocity_scale**2
gamma = 100.0

# let's just do this at a fixed timestep
timestep = 1.1

def kinetic_energy(v):
    return 0.5 * m * v**2


def get_sampled_joint_histogram(x, v):
    joint_sampled_dist, _, _ = np.histogram2d(x, v, bins=bin_edges, normed=True)
    return joint_sampled_dist


def get_sampled_x_histogram(x):
    sampled_x_hist, bin_edges = np.histogram(x, **one_d_hist_args)
    return sampled_x_hist, bin_edges


def get_hists(integrator, n_steps_per_batch=1000000, n_batches=100):
    x_0, v_0 = np.random.randn(2)
    # just a quick warm-up
    xs, vs, Q, W_shads = integrator(x_0, v_0, n_steps_per_batch, gamma, timestep)
    joint_hist = get_sampled_joint_histogram(xs[:, 0], vs[:, 0])
    joint_hists = [0 * joint_hist] * xs.shape[1]
    n_hists = xs.shape[1]

    x_hist, x_bin_edges = get_sampled_x_histogram(xs[:, 0])
    x_hists = [0 * x_hist] * n_hists
    x_0 = xs[-1][-1]
    v_0 = vs[-1][-1]

    print("Simulating for {} steps".format(n_batches * n_steps_per_batch))
    # To avoid explosion of memory consumption, update the histograms in batches

    for _ in tqdm(range(n_batches)):
        xs, vs, Q, W_shads = integrator(x_0, v_0, n_steps_per_batch, gamma, timestep)

        for i in range(n_hists):
            joint_hists[i] = joint_hists[i] + get_sampled_joint_histogram(xs[:, i], vs[:, i]) / n_hists
            x_hists[i] = x_hists[i] + get_sampled_x_histogram(xs[:, i])[0] / n_hists

        x_0 = xs[-1][-1]
        v_0 = vs[-1][-1]

    return joint_hists, x_hists

def simulate_all():
    vvvr = vvvr_factory(potential, force, velocity_scale, m)
    vvvr_joint_hists, vvvr_x_hists = get_hists(vvvr)

    for i in range(5):
        np.save("vvvr_joint_hist_{}.npy".format(i), vvvr_joint_hists[i])
        np.save("vvvr_x_hist_{}.npy".format(i), vvvr_x_hists[i])

    baoab = baoab_factory(potential, force, velocity_scale, m)
    baoab_joint_hists, baoab_x_hists = get_hists(baoab)

    for i in range(5):
        np.save("baoab_joint_hist_{}.npy".format(i), baoab_joint_hists[i])
        np.save("baoab_x_hist_{}.npy".format(i), baoab_x_hists[i])

if __name__ == "__main__":
    simulate_all()
    plot_all()

