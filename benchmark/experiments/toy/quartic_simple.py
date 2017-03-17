# Simplest case, no bells or whistles, just hard-code integrators and test system

import numpy as np
import matplotlib
from numba import jit
from time import time

matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy

# define system
np.random.seed(0)

figure_directory = "figures/"  # relative to script
figure_format = ".pdf"

# Define system
beta = 1.0  # inverse temperature
dim = 1  # system dimension

@jit
def quartic_potential(x): return x**4

@jit
def quartic_force(x): return - 4.0 * x**3


@jit
def eggcrate_potential(x): return x**2 + np.sin(10 * x)

@jit
def eggcrate_force(x): return - ( 2 * ( x + 5 * np.cos(10 * x)) )


system = "quartic"

if system == "quartic":
    potential = quartic_potential
    force = quartic_force
elif system == "eggcrate":
    potential = eggcrate_potential
    force = eggcrate_force

@jit
def reduced_potential(x): return potential(x) * beta


@jit
def log_q(x): return - reduced_potential(x)


@jit
def q(x): return np.exp(log_q(x))


# normalized density
x = np.linspace(-3, 3, 1000)
x_ = np.linspace(-10,10,10000)
Z = np.trapz(q(x_))
log_Z = np.log(Z)

def p(x): return q(x) / Z
def log_p(x): return log_q(x) - log_Z

# example initial conditions
x_0, v_0 = np.random.randn(), np.random.randn()

m = 10.0  # mass
velocity_scale = np.sqrt(1.0 / (beta * m))
sigma2 = velocity_scale**2
timestep = 1.0
gamma = 100.0

# implement ovrvo
def simulate_vvvr(x0, v0, n_steps, gamma, dt, thinning_factor=1):
    """Simulate n_steps of VVVR, accumulating heat

    :param x0:
    :param v0:
    :param n_steps:
    :param gamma:
    :param dt:
    :return:
    """
    Q = 0
    W_shads = np.zeros(n_steps / thinning_factor)
    x, v = x0, v0
    xs, vs = np.zeros(n_steps / thinning_factor), np.zeros(n_steps / thinning_factor)
    xs[0] = x0
    vs[0] = v0
    E_old = potential(x) + 0.5 * m * v**2

    a = np.exp(-gamma * (dt / 2.0))
    b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

    for i in range(1, n_steps):
        # O step
        ke_old = 0.5 * m * v**2
        v = (a * v) + b * velocity_scale * np.random.randn()
        ke_new = 0.5 * m * v ** 2
        Q += (ke_new - ke_old)

        # V step
        v = v + ((dt / 2.0) * force(x) / m)

        # R step
        x = x + (dt * v)

        # V step
        v = v + ((dt / 2.0) * force(x) / m)

        # O step
        ke_old = 0.5 * m * v ** 2
        v = (a * v) + b * velocity_scale * np.random.randn()
        ke_new = 0.5 * m * v ** 2
        Q += (ke_new - ke_old)

        # store
        if i % thinning_factor == 0:
            xs[i / thinning_factor] = x
            vs[i / thinning_factor] = v

            E_new = potential(x) + 0.5 * m * v ** 2
            W_shads[i / thinning_factor] = (E_new - E_old) - Q

    return xs, vs, Q, W_shads


# implement BAOAB / VRORV
def simulate_baoab(x0, v0, n_steps, gamma, dt, thinning_factor=1):
    """Simulate n_steps of BAOAB, accumulating heat

    :param x0:
    :param v0:
    :param n_steps:
    :param gamma:
    :param dt:
    :return:
    """
    Q = 0
    W_shads = np.zeros(n_steps / thinning_factor)
    x, v = x0, v0
    xs, vs = np.zeros(n_steps / thinning_factor), np.zeros(n_steps / thinning_factor)
    xs[0] = x0
    vs[0] = v0
    E_old = potential(x) + 0.5 * m * v**2

    a = np.exp(-gamma * (dt))
    b = np.sqrt(1 - np.exp(-2 * gamma * (dt)))

    for i in range(1, n_steps):
        # V step
        v = v + ((dt / 2.0) * force(x) / m)

        # R step
        x = x + ((dt / 2.0) * v)

        # O step
        ke_old = 0.5 * m * v**2
        v = (a * v) + b * velocity_scale * np.random.randn()
        ke_new = 0.5 * m * v ** 2
        Q += (ke_new - ke_old)

        # R step
        x = x + ((dt / 2.0) * v)

        # V step
        v = v + ((dt / 2.0) * force(x) / m)

        # store
        if i % thinning_factor == 0:
            xs[i / thinning_factor] = x
            vs[i / thinning_factor] = v

            E_new = potential(x) + 0.5 * m * v**2
            W_shads[i / thinning_factor] = (E_new - E_old) - Q

    return xs, vs, Q, W_shads

def rw_metropolis_hastings(x0, n_steps):
    xs = np.zeros(n_steps)
    xs[0] = x0

    # draw all the random numbers we'll need
    proposal_eps = np.random.randn(n_steps) # standard normal
    accept_eps = np.random.rand(n_steps) # uniform(0,1)

    for i in range(1, n_steps):
        x_prop = xs[i-1] + proposal_eps[i]
        a_r_ratio = q(x_prop) / q(xs[i-1])

        # accept / reject
        if a_r_ratio > accept_eps[i]:
            xs[i] = x_prop
        else:
            xs[i] = xs[i-1]
    return xs

# also jit compile...
fast_simulate = jit(simulate_vvvr)
_ = fast_simulate(0.0, 0.0, 10, 10.0, 1.0)

fast_simulate_baoab = jit(simulate_baoab)
_ = fast_simulate_baoab(0.0, 0.0, 10, 10.0, 1.0)

fast_mh = jit(rw_metropolis_hastings)
_ = fast_mh(0.0, 10)

def speed_test(n_steps=100000):
    t0 = time()
    _ = simulate_vvvr(x_0, v_0, n_steps, gamma, timestep)
    t1 = time()

    t2 = time()
    _ = fast_simulate(x_0, v_0, n_steps, gamma, timestep)
    t3 = time()

    print("Time to take {} steps, Python: {:.5f}s".format(n_steps, t1 - t0))
    print("Time to take {} steps, JIT'd: {:.5f}s".format(n_steps, t3 - t2))
    print("Factor improvement: {:.3f}x".format((t1 - t0) / (t3 - t2)))

    t0 = time()
    _ = rw_metropolis_hastings(0, n_steps)
    t1 = time()

    t2 = time()
    _ = fast_mh(0, n_steps)
    t3 = time()

    print("Time to take {} MH steps, Python: {:.5f}s".format(n_steps, t1 - t0))
    print("Time to take {} MH steps, JIT'd: {:.5f}s".format(n_steps, t3 - t2))
    print("Factor improvement: {:.3f}x".format((t1 - t0) / (t3 - t2)))

def compute_free_energy_potential_and_entropy(x_samples, hist_args):
    # print average potential energy
    avg_potential = np.mean(potential(x_samples))
    print("\t<U> = {:.5f}".format(avg_potential))

    # now, what's the entropy
    hist, bin_edges = np.histogram(x_samples, **hist_args)
    #hist, _ = np.histogram(x_samples, bins="auto")
    ent = entropy(hist, base=np.e) / (bin_edges[-1] - bin_edges[0])
    print("\tS = {:.5f} (using {}-bin histogram)".format(ent, len(hist)))

    return avg_potential - ent / beta



@jit
def estimate_Delta_F_neq_conf_vvvr(x_samples, gamma, dt, protocol_length=100, n_samples=1000):

    # indices of samples drawn with replacement from initial equilibrium samples
    selections = np.random.randint(0, len(x_samples), n_samples)
    velocities = np.random.randn(n_samples, 2) * velocity_scale

    W_shads_F = np.zeros((n_samples, protocol_length))
    W_shads_R = np.zeros((n_samples, protocol_length))

    for i in range(n_samples):
        x0 = x_samples[selections[i]]
        v0 = velocities[i, 0]

        xs, vs, Q, W_shads = fast_simulate(x0, v0, protocol_length, gamma, dt)
        W_shads_F[i] = W_shads

        x1 = xs[-1]
        v1 = velocities[i, 1]

        xs, vs, Q, W_shads = fast_simulate(x1, v1, protocol_length, gamma, dt)

        W_shads_R[i] = W_shads

    return W_shads_F, W_shads_R

@jit
def estimate_Delta_F_neq_conf_baoab(x_samples, gamma, dt, protocol_length=100, n_samples=1000):

    # indices of samples drawn with replacement from initial equilibrium samples
    selections = np.random.randint(0, len(x_samples), n_samples)
    velocities = np.random.randn(n_samples, 2) * velocity_scale

    W_shads_F = np.zeros((n_samples, protocol_length))
    W_shads_R = np.zeros((n_samples, protocol_length))

    for i in range(n_samples):
        x0 = x_samples[selections[i]]
        v0 = velocities[i, 0]

        xs, vs, Q, W_shads = fast_simulate_baoab(x0, v0, protocol_length, gamma, dt)
        W_shads_F[i] = W_shads

        x1 = xs[-1]
        v1 = velocities[i, 1]

        xs, vs, Q, W_shads = fast_simulate_baoab(x1, v1, protocol_length, gamma, dt)

        W_shads_R[i] = W_shads

    return W_shads_F, W_shads_R

def log_v_density(v):
    return -v ** 2 / (2 * sigma2) - np.log((np.sqrt(2 * np.pi * sigma2)))

def v_density(v):
    return np.exp(log_v_density(v))

def normalize_histogram(hist, bin_edges):
    x_range = bin_edges[-1] - bin_edges[0]
    sum_y = np.sum(hist)
    Z = (sum_y / x_range)
    return hist / Z

def compute_exact_histogram(density, bin_edges):
    exact_hist = np.zeros(len(eq_hist))
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        x_ = np.linspace(left, right, 1000)
        y_ = density(x_)
        exact_hist[i] = np.trapz(y_, x_)
    # let's double-check to make sure this histogram is normalized
    return normalize_histogram(exact_hist, bin_edges)

def plot_difference_between_histograms(data, exact_hist, range, n_bins=100):
    hist_args = {"bins": n_bins, "range": range, "density": True}
    hist, bin_edges = np.histogram(data, **hist_args)
    hist = normalize_histogram(hist, bin_edges)

    bin_width = (bin_edges[1] - bin_edges[0]) / 2
    x_points = bin_edges[1:] - bin_width

    plt.plot(x_points, exact_hist, label="Exact", c="blue")
    plt.plot(x_points, hist, label="Sampled", c="red")
    plt.fill_between(x_points, hist, exact_hist, alpha=0.4, color="grey", label="Difference")

    plt.legend(loc="best", fancybox=True)

if __name__ == "__main__":
    np.random.seed(12345)

    # now, collect a bunch of samples, compute histograms
    n_steps = 100000000
    thinning_factor = 5

    # generate plots
    left, right = -3, 3
    x = np.linspace(left, right, 1000)
    v_p = v_density(x)

    histstyle = {"bins" : 200,
                 "normed" : True,
                 "histtype" : "stepfilled",
                 "alpha" : 0.5}

    # what's the average potential energy
    avg_potential = np.trapz(map(lambda x:q(x)*potential(x), x), x)

    # let's collect some equilibrium samples
    eq_xs = fast_mh(0.0, n_steps)
    hist_args = {"bins": 200, "range": (left, right)}
    eq_hist, _ = np.histogram(eq_xs, **hist_args)
    print("Equilibrium samples:")
    F_eq = compute_free_energy_potential_and_entropy(eq_xs, hist_args)

    # let's also compute the equilibrium histogram ~exactly
    # (this is important because the KL divergence between the raw sample
    # histograms was often inf, due to no equilibrium samples in the extreme tails)
    eq_hist, bin_edges = np.histogram(eq_xs, **hist_args)
    exact_eq_hist = compute_exact_histogram(q, bin_edges)
    exact_v_hist = compute_exact_histogram(v_density, bin_edges)

    plt.figure()
    plt.plot(exact_eq_hist)
    plt.savefig("{}exact_eq_hist{}".format(figure_directory, figure_format))
    plt.close()

    print("D_KL between sampled eq_hist and exact_eq_hist: {:.5f}".format(
          entropy(eq_hist, exact_eq_hist)))

    eq_hist = exact_eq_hist

    plt.figure()
    plt.hist(eq_xs, **histstyle)  # histogram of x samples
    plt.plot(x, map(p, x))  # actual density
    plt.savefig("{}x_samples_equil{}".format(figure_directory, figure_format), dpi=300)
    plt.close()

    def compare_estimators(scheme="VVVR"):
        KLs_direct = []
        KLs_hist = []
        KLs_prot = []
        KLs_prot_err = []

        timesteps_to_try = np.array([0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])

        for i, dt in enumerate(timesteps_to_try):
            print(dt)
            if scheme == "VVVR":
                xs, vs, Q, W_shad = fast_simulate(x_0, v_0, n_steps, gamma, dt, thinning_factor)
            elif scheme == "BAOAB":
                xs, vs, Q, W_shad = fast_simulate_baoab(x_0, v_0, n_steps, gamma, dt, thinning_factor)
            name = "{}_dt={}".format(scheme, dt).replace('.','-') + figure_format
            xs = xs[100:]
            vs = vs[100:]

            if len(np.where(np.isnan(xs))[0]) > 0:
                max_ind = np.where(np.isnan(xs))[0][0]
                print("\tNaN encountered after {} steps!".format(thinning_factor * max_ind))

                xs = xs[:max_ind - 10]
                vs = vs[:max_ind - 10]

            print("\nTesting {} with timestep dt={}".format(scheme, dt))

            # plot x histogram
            plt.figure()
            plot_difference_between_histograms(xs, exact_eq_hist, range=(left, right), n_bins=hist_args["bins"])
            plt.xlabel("Configuration ($x$)")
            plt.ylabel("Probability density")
            plt.yticks([])
            plt.savefig("{}x_samples_{}".format(figure_directory, name), dpi=300)
            plt.close()

            # plot v histogram
            plt.figure()
            plot_difference_between_histograms(vs, exact_v_hist, range=(left, right), n_bins=hist_args["bins"])
            plt.xlabel("Velocity ($v$)")
            plt.ylabel("Probability density")
            plt.yticks([])
            plt.savefig("{}v_samples_{}".format(figure_directory, name), dpi=300)
            plt.close()

            F_neq = compute_free_energy_potential_and_entropy(xs, hist_args)

            noneq_hist, _ = np.histogram(xs, **hist_args)

            # print KL divergence estimated a few different ways
            KLs_direct.append(F_neq - F_eq)
            KLs_hist.append(entropy(noneq_hist, eq_hist))
            print("\tDelta F_neq where F = <E> - entropy(histogram) / beta : {:.5f}".format(KLs_direct[-1]))
            print("\tHistogram D_KL(p_neq(x) || p_eq(x)) : {:.5f}".format(KLs_hist[-1]))

            # compute conf-space Delta F_neq
            protocol_length = 100
            n_protocol_samples = 2000000
            if scheme=="VVVR":
                W_shads_F, W_shads_R = estimate_Delta_F_neq_conf_vvvr(eq_xs, gamma, dt, protocol_length, n_protocol_samples)
            elif scheme=="BAOAB":
                W_shads_F, W_shads_R = estimate_Delta_F_neq_conf_baoab(eq_xs, gamma, dt, protocol_length, n_protocol_samples)

            W_F = W_shads_F[:, -1]
            W_R = W_shads_R[:, -1]
            N = len(W_F)
            DeltaF_neq = 0.5 * (np.mean(W_F) - np.mean(W_R))
            sq_uncertainty = (np.var(W_F) + np.var(W_R) - 2 * np.cov(W_F, W_R)[0, 1]) / (4 * N)
            err = 1.96 * np.sqrt(sq_uncertainty)

            KLs_prot.append(DeltaF_neq)
            KLs_prot_err.append(err)

            print("\tNear-eq approx Delta F_neq : {:.5f} +/- {:.5f}".format(DeltaF_neq, err))
            # now, also plot the work distributions
            plt.figure()
            plt.hist(W_F, bins=50, alpha=0.5, normed=True, label=r"$W_{\pi \to \rho} \stackrel{?}{=} W_{\pi \to \omega}$")
            plt.hist(W_R, bins=50, alpha=0.5, normed=True, label=r"$W_{\omega \to \rho}$")
            plt.legend(loc="best", fancybox=True)
            plt.xlabel("Work")
            plt.ylabel("Probability density")
            plt.yscale("log")
            plt.title("{}, dt={}: Nonequilibrium work distributions".format(scheme, dt))
            plt.savefig("{}work_dists_{}".format(figure_directory, name), dpi=300)
            plt.close()

            # to-do: also plot the work trajectories, so that we can verify that we're in steady state
            mean_F = np.mean(W_shads_F, 0)
            mean_R = np.mean(W_shads_R, 0)
            err_F = 1.96 * np.std(W_shads_F, 0) / np.sqrt(len(W_shads_F))
            err_R = 1.96 * np.std(W_shads_R, 0) / np.sqrt(len(W_shads_R))

            plt.figure()
            ax = plt.subplot(121)
            ax.plot(mean_F, c="blue", linewidth=3)
            ax.fill_between(range(len(mean_F)), mean_F - err_F, mean_F + err_F, color="blue", alpha=0.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Work")
            ax.set_title(r"$W_{\pi \to \rho}$")

            ax1 = plt.subplot(122, sharey=ax)
            ax1.plot(mean_R, c="red", linewidth=3)
            ax1.fill_between(range(len(mean_F)), mean_R - err_R, mean_R + err_R, color="red", alpha=0.5)

            ax1.set_xlabel("Step")
            ax1.set_title(r"$W_{\omega \to \rho}$")

            plt.savefig("{}work_trajs_{}".format(figure_directory, name), dpi=300)
            plt.close()


        # plot the various estimates

        #plt.plot(timesteps_to_try, KLs_direct, label=r"{} $(\langle E \rangle_{neq} - S_{neq}) - (\langle E \rangle_{eq} - S_{eq})$")
        if scheme == "VVVR":
            label = "VVVR: $D_{KL}(p_{neq}(x) \| p_{eq}(x))$"
        elif scheme == "BAOAB":
            label = "BAOAB: $D_{KL}(p_{neq}(x) \| p_{eq}(x))$"
        plt.plot(timesteps_to_try, KLs_hist, label=label)

        plt.errorbar(timesteps_to_try, KLs_prot, KLs_prot_err, label="{}: noneq estimate".format(scheme))


    plt.figure()
    compare_estimators("VVVR")
    compare_estimators("BAOAB")
    plt.xlabel("Timestep")
    plt.ylabel("KL divergence")
    plt.title("Validating noneq estimator of the timestep-dependent\n"
              "configuration-space error on 1D quartic potential")
    plt.legend(loc="best", fancybox=True)
    plt.savefig("{}estimator_comparison{}".format(figure_directory, figure_format), dpi=300)
    plt.close()


    # next things:
    # * plot contours of 2D joint distribution over (x,v)
    # * add plot of total KL divergence for each vs. estimated total KL divergence?