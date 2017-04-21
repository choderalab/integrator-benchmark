"""Pure python example on quartic potential, for illustration and to identify
implementation errors / pitfalls that occur independent of the complexity of
OpenMM (numerical precision, constraints, CustomIntegrator compilation, etc.).

Current problems:
* Shadow work in first step appears to be negative for some integrator schemes
  (OVRVO and RVOVR)
* Weird jump in accumulated shadow work after first step, even when using
  null midpoint operator (for all schemes). For the null midpoint operator,
  it should be in steady state / the first step shouldn't be statistically
  any different from any of the following steps...
"""

import numpy as np
import matplotlib
from numba import jit

matplotlib.use('agg')
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import entropy
import statsmodels.api as sm

# import pymbar
np.random.seed(0)

figure_directory = "figures/"  # relative to script
figure_format = ".jpg"

# Define system
beta = 1.0  # inverse temperature
dim = 1  # system dimension


def potential(x):
    return x**4


def reduced_potential(x): return potential(x) * beta


@jit
def log_q(x): return - reduced_potential(x)


@jit
def q(x): return np.exp(log_q(x))


def force(x):
    return - 4.0 * x ** 3


# normalized density
x = np.linspace(-3, 3, 1000)
Z = np.trapz(map(q, x), x)


def p(x): return q(x) / Z


def plot(x, y, color='blue', padding=0):
    plt.plot(x, y, alpha=0.5, linewidth=2, color=color)
    plt.fill_between(x, y, alpha=0.25, color=color)
    span = max(y) - min(y)
    plt.ylim(min(y) - (padding * span), max(y) + (padding * span))


def savefig(name):
    plt.savefig("{}{}{}".format(figure_directory, name, figure_format), dpi=300)


def plot_system():
    print("Plotting system...")

    plt.figure()
    plt.subplot(1, 3, 1)
    plot(x, map(reduced_potential, x))
    plt.title('Reduced potential\n$u(x)=\\beta U(x)$')

    plt.subplot(1, 3, 2)
    plot(x, map(p, x))
    plt.title('Equilibrium\ndistribution\n$p(x)\propto e^{-u(x)}$')

    plt.subplot(1, 3, 3)
    plot(x, map(force, x))
    plt.title('Force\n$F(x)=-\\nabla U(x)$')

    plt.tight_layout()
    savefig("quartic_system")
    plt.close()

# example initial conditions
x_0, v_0 = np.zeros(dim), np.ones(dim)

m = 10.0  # mass
velocity_scale = np.sqrt(1.0 / (beta * m))
timestep = 1.0
gamma = 100.0


def draw_velocities():
    return velocity_scale * np.random.randn(dim)


@jit
def kinetic_energy(v):
    return np.sum(0.5 * m * v ** 2)


@jit
def total_energy(x, v):
    return kinetic_energy(v) + potential(x)


@jit
def R_map(x, v, h):  # linear "drift"
    x_ = x + (h * v)
    v_ = v
    return x_, v_


@jit
def V_map(x, v, h):  # linear "kick"
    x_ = x
    v_ = v + (h * force(x) / m)
    return x_, v_


@jit
def O_map(x, v, h, gamma):  # Ornstein-Uhlenbeck
    x_ = x
    a = np.exp(-gamma * h)
    b = np.sqrt(1 - np.exp(-2 * gamma * h))
    v_ = (a * v) + b * draw_velocities()
    return x_, v_


splitting_map = {'R': R_map, 'V': V_map, 'O': O_map}


@jit
def langevin_map(x, v, h, splitting, gamma):
    n_R = sum([l == "R" for l in splitting])
    n_V = sum([l == "V" for l in splitting])
    n_O = sum([l == "O" for l in splitting])

    for step in splitting:
        if step == "R":
            x, v = R_map(x, v, h / n_R)
        elif step == "V":
            x, v = V_map(x, v, h / n_V)
        elif step == "O":
            x, v = O_map(x, v, h / n_O, gamma)
    return x, v


# now, we just wrap each of the R, V, O updates with a couple lines to measure
# the energy difference during the substep
@jit
def bookkeeping_R_map(x, v, h):
    ''' deterministic position update '''
    x_, v_ = R_map(x, v, h)
    e_diff = potential(x_) - potential(x)
    return x_, v_, e_diff


@jit
def bookkeeping_V_map(x, v, h):
    ''' deterministic velocity update '''
    x_, v_ = V_map(x, v, h)
    e_diff = kinetic_energy(v_) - kinetic_energy(v)
    return x_, v_, e_diff


@jit
def bookkeeping_O_map(x, v, h, gamma):
    ''' stochastic velocity update'''
    x_, v_ = O_map(x, v, h, gamma)
    e_diff = kinetic_energy(v_) - kinetic_energy(v)
    return x_, v_, e_diff


bookkeeping_map = {'R': bookkeeping_R_map,
                   'V': bookkeeping_V_map,
                   'O': bookkeeping_O_map}


def bookkeeping_langevin_map(x, v, h, splitting, gamma):
    W_shad, Q = 0, 0
    n_R = sum([l == "R" for l in splitting])
    n_V = sum([l == "V" for l in splitting])
    n_O = sum([l == "O" for l in splitting])

    for i in range(len(splitting)):
        substep = splitting[i]

        if substep == "R":
            h_substep = h / n_R
            x, v, e_diff = bookkeeping_R_map(x, v, h_substep)

        elif substep == "V":
            h_substep = h / n_V
            x, v, e_diff = bookkeeping_V_map(x, v, h_substep)

        elif substep == "O":
            h_substep = h / n_O
            x, v, e_diff = bookkeeping_O_map(x, v, h_substep, gamma)

        if substep in "RV":
            W_shad += e_diff
        else:
            Q += e_diff

    return x, v, W_shad, Q


def bookkeeping_langevin_factory(n_steps, splitting='VRORV'):
    def multistep_langevin(x, v, h, gamma):
        xs, vs = np.zeros((n_steps, len(x))), np.zeros((n_steps, len(x)))
        xs[0], xs[0] = x, v
        W_shad, Q = np.zeros(n_steps), np.zeros(n_steps)
        for i in range(1, n_steps):
            xs[i], vs[i], W_shad[i], Q[i] = \
                bookkeeping_langevin_map(xs[i - 1], vs[i - 1], h, splitting, gamma)
        return xs, vs, W_shad, Q

    return multistep_langevin


def test_scheme(x_0, v_0, n_steps, scheme, h, gamma):
    langevin = bookkeeping_langevin_factory(n_steps, scheme)
    xs, vs, W_shad, Q = langevin(x_0, v_0, h, gamma)

    # plot the x marginal
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Sampled x ({})'.format(scheme))
    plt.hist(xs.flatten(), bins=50, normed=True, histtype='stepfilled', alpha=0.5);
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, map(p, x))
    plt.xlabel(r'$x$')
    plt.ylabel('Sampled P(x)')

    # plot the trajectory
    plt.subplot(1, 2, 2)
    plt.title('Trajectory ({})'.format(scheme))
    plt.plot(xs[:, 0], vs[:, 0] * m, linewidth=0.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$mv$')

    plt.tight_layout()

    savefig("quartic_samples_{}".format(scheme))
    plt.close()

    # plot the accumulated shadow work
    plt.figure()
    plt.plot(np.cumsum(W_shad))
    plt.xlabel('Step')
    plt.ylabel('Accumulated W_shad')
    plt.title('Accumulated shadow work ({})'.format(scheme))
    savefig("quartic_shadow_work_{}".format(scheme))
    plt.close()

    return xs, vs, W_shad, Q

#schemes = "VRORV RVOVR OVRVO".split()
#results = dict()
#print("Simulating long trajectories using {} integrators".format(schemes))
#for scheme in schemes:
#    results[scheme] = test_scheme(np.random.randn(dim), draw_velocities(),
#                                  n_steps=int(1e4), scheme=scheme, h=timestep,
#                                  gamma=gamma
#                                  )

# let's confirm that energy change accumulated by the bookkeeper is the same as the
# actual energy change
#print("Confirming that bookkeeping is self-consistent...")


def check_bookkeeping(results):
    for scheme in results:
        xs, vs, W_shad, Q = results[scheme]
        es = np.array([total_energy(xs[i], vs[i]) for i in range(len(xs))])

        # accumulated
        delta_e_bk = np.cumsum(W_shad + Q)
        accumulated = delta_e_bk - delta_e_bk[0]

        actual = es - es[0]
        discrepancy = accumulated - actual

        print('\t{}: Mean squared discrepancy between '
              'accumulated and actual energy change: {:.5f}'.format(scheme, np.mean(np.linalg.norm(discrepancy))))


#check_bookkeeping(results)

### Estimate nonequilibrium free energy difference
def collect_equilibrium_samples():
    print("Collecting equilibrium samples...")
    # collect equilibrium samples
    import emcee

    n_walkers = 2 * dim
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=dim, lnpostfn=log_q)
    sampler.run_mcmc(pos0=np.random.randn(n_walkers, dim), N=50000)
    equilibrium_samples = sampler.flatchain[1000:]
    draw_configuration = lambda: equilibrium_samples[np.random.randint(len(equilibrium_samples))]
    plt.figure()
    plt.title('Equilibrium samples')
    plt.hist(equilibrium_samples.flatten(), bins=50, normed=True, histtype='stepfilled', alpha=0.5);
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, map(p, x))
    plt.xlabel(r'$x$')
    plt.ylabel('Sampled P(x)')
    savefig("quartic_equilibrium_samples")
    plt.close()


def run_protocol(integrator, midpoint_operator):
    # simulate starting from equilibrium sample
    x_traj, v_traj, W_shad, Q = integrator(draw_configuration(), draw_velocities())
    W_shad_F = np.cumsum(W_shad)

    # apply midpoint operator
    if midpoint_operator == "null":
        x_1, v_1 = x_traj[-1], v_traj[-1]
    elif midpoint_operator == "randomize-velocity":
        x_1, v_1 = x_traj[-1], draw_velocities()
    elif midpoint_operator == "randomize-positions":
        x_1, v_1 = draw_configuration(), v_traj[-1]
    else:
        raise (ValueError("Invalid midpoint operator"))

    # simulate starting from out-of-equilibrium ensemble
    _, _, W_shad, _ = integrator(x_1, v_1)
    W_shad_R = np.cumsum(W_shad)

    return W_shad_F, W_shad_R, x_1


def perform_benchmark(integrator, midpoint_operator, n_samples):
    samples = [run_protocol(integrator, midpoint_operator) for _ in range(n_samples)]
    W_shads_F, W_shads_R, x = [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]
    return W_shads_F, W_shads_R, x


def estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R, N_eff=None, verbose=True):
    if N_eff == None: N_eff = len(W_shads_F)
    DeltaF_neq = 0.5 * (np.mean(W_shads_F) - np.mean(W_shads_R))
    # squared uncertainty = [var(W_0->M) + var(W_M->2M) - 2 cov(W_0->M, W_M->2M)] / (4 N_eff)
    sq_uncertainty = (np.var(W_shads_F) + np.var(W_shads_R) - 2 * np.cov(W_shads_F, W_shads_R)[0, 1]) / (4 * N_eff)
    if verbose:
        print(
        "\t<W_F>: {:.5f} +/- {:.5f}".format(np.mean(W_shads_F), 1.96 * np.std(W_shads_F) / np.sqrt(len(W_shads_F))))
        print(
        "\t<W_R>: {:.5f} +/- {:.5f}".format(np.mean(W_shads_R), 1.96 * np.std(W_shads_R) / np.sqrt(len(W_shads_R))))
        print("\tcov(W_F, W_R): {:.5f}".format(np.cov(W_shads_F, W_shads_R)[0, 1]))
    return DeltaF_neq, sq_uncertainty


## KDE-based, via <U> - S
def estimate_entropy_kde_1d(samples):
    dens = sm.nonparametric.KDEUnivariate(samples)
    dens.fit()
    return dens.entropy


def estimate_F(X, U):
    '''

    Define free energy, given an energy function U and samples
     from P(x) = 1/Z e^{-U(x)} as the
    average energy - the entropy:

    F_P = \langle U \rangle_P - S_P

    Parameters
    ----------
    X : samples
    U : reduced potential function

    Returns
    -------
    F_hat : <U> - S
    '''

    # compute average energy
    U_X = np.array([U(x) for x in X])
    mean_U = np.mean(U_X)

    # compute entropy
    S = estimate_entropy_kde_1d(X)

    return mean_U - S


def estimate_dFNeq_kde(X_neq, X_eq, U):
    return estimate_F(X_neq, U) - estimate_F(X_eq, U)


## work-based


# def BAR_estimator(W_shad, W_ss):
#    return pymbar.BAR(W_shad,W_ss)[0]

def plot_mean_and_95_confidence_bands(array, x=None, color="blue", label=None):
    """Given a list of trajectories, plot the mean trajectory +/- 1.96 * standard error"""
    mean = np.mean(array, 0)
    band = 1.96 * np.std(array, 0) / np.sqrt(len(array))

    plt.plot(mean, color=color, label=label)
    if x == None: x = range(len(array[0]))
    plt.fill_between(x, mean - band, mean + band, color=color, alpha=0.3)


def unpack_W_shads(W_shads_F, W_shads_R):
    return [F[-1] for F in W_shads_F], [R[-1] for R in W_shads_R]


# okay, now we should also do the comparison -- apply each estimator

if __name__ == "__main__":
    n_steps = 20
    results = dict()
    midpoint_operators = ["null", "randomize-velocity", "randomize-positions"]
    colors = {"null": "blue",
              "randomize-velocity": "green",
              "randomize-positions": "orange"
              }

    for scheme in schemes:
        print("\n\nScheme: {}".format(scheme))

        integrator = partial(bookkeeping_langevin_factory(n_steps, scheme), h=timestep, gamma=gamma)

        results[scheme] = dict()
        plt.figure()
        for midpoint_operator in midpoint_operators:
            print("\nMidpoint operator: {}".format(midpoint_operator))
            W_shads_F, W_shads_R, x = perform_benchmark(integrator, midpoint_operator, n_samples=50000)
            results[scheme][midpoint_operator] = W_shads_F, W_shads_R, x

            ax1 = plt.subplot(1, 2, 1)
            plot_mean_and_95_confidence_bands(W_shads_F, color=colors[midpoint_operator], label=midpoint_operator)
            plt.title("{} for {} steps,\nstarting in equilibrium".format(scheme, len(W_shads_F[0])))
            plt.xlabel('steps')
            plt.ylabel('W_shad')
            ax2 = plt.subplot(1, 2, 2, sharey=ax1)
            plot_mean_and_95_confidence_bands(W_shads_R[1:], color=colors[midpoint_operator], label=midpoint_operator)

            title = "{} for another {} steps".format(scheme, len(W_shads_R[0]))
            if midpoint_operator != "null":
                title += ",\n after applying midpoint operator"
            plt.title(title)
            plt.xlabel('steps')

            DeltaF_neq, sq_uncertainty = estimate_nonequilibrium_free_energy(*unpack_W_shads(W_shads_F, W_shads_R))

            plt.legend(fancybox=True, loc='best')

            print("\tDeltaF_neq: {:.5f} +/- {:.5f}".format(DeltaF_neq, np.sqrt(sq_uncertainty)))

        savefig("benchmark_{}".format(scheme))
        plt.close()

        # plot midprotocol x samples
        x_samples = np.vstack([results[scheme][midpoint_operator][-1] for midpoint_operator in midpoint_operators])
        plt.figure()
        plt.title('Sampled x ({})'.format(scheme))
        plt.hist(x_samples.flatten(), bins=50, normed=True, histtype='stepfilled', alpha=0.5);
        x = np.linspace(-3, 3, 1000)
        plt.plot(x, map(p, x))
        plt.xlabel(r'$x$')
        plt.ylabel('Sampled P(x)')
        savefig("quartic_midprotocol_samples_{}".format(scheme))
        plt.close()

        # let's also do the comparison in section 5 of https://arxiv.org/pdf/1203.5428.pdf
