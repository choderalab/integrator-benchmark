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
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
np.random.seed(0)

figure_directory = "figures/" # relative to script
figure_format = ".jpg"

# Define system
beta = 1.0 # inverse temperature
dim = 1 # system dimension

def potential(x): return np.sum(x**4)
def reduced_potential(x): return potential(x) * beta
def log_q(x): return - reduced_potential(x)
def q(x): return np.exp(log_q(x))
def force(x): return - 4.0 * np.sum(x**3)

# normalized density
x = np.linspace(-3,3,1000)
Z = np.trapz(map(q, x), x)
def p(x): return q(x) / Z

def plot(x, y, color='blue', padding=0):
    plt.plot(x, y, alpha=0.5, linewidth=2,color=color)
    plt.fill_between(x,y,alpha=0.25,color=color)
    span = max(y) - min(y)
    plt.ylim(min(y) - (padding * span), max(y) + (padding * span))

def savefig(name):
    plt.savefig("{}{}{}".format(figure_directory, name, figure_format), dpi=300)

print("Plotting system...")

plt.figure()
plt.subplot(1,3,1)
plot(x, map(reduced_potential, x))
plt.title('Reduced potential\n$u(x)=\\beta U(x)$')

plt.subplot(1,3,2)
plot(x, map(p, x))
plt.title('Equilibrium\ndistribution\n$p(x)\propto e^{-u(x)}$')

plt.subplot(1,3,3)
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
integrator_params = {'gamma': 100.0}


def draw_velocities():
    return velocity_scale * np.random.randn(dim)

def kinetic_energy(v):
    return np.sum(0.5 * m * v ** 2)


def total_energy(x, v):
    return kinetic_energy(v) + potential(x)


def R_map(x, v, h):  # linear "drift"
    x_ = x + (h * v)
    v_ = v
    return x_, v_


def V_map(x, v, h):  # linear "kick"
    x_ = x
    v_ = v + (h * force(x) / m)
    return x_, v_


def O_map(x, v, h, gamma):  # Ornstein-Uhlenbeck
    x_ = x
    b = np.exp(-gamma * h)
    v_ = (b * v) + (np.sqrt(1 - b ** 2) * draw_velocities())
    return x_, v_


splitting_map = {'R': R_map, 'V': V_map, 'O': O_map}


def langevin_map(x, v, h, splitting):
    for step in splitting: x, v = splitting_map[step](x, v, h / sum([l == step for l in splitting]))
    return x, v

# now, we just wrap each of the R, V, O updates with a couple lines to measure
# the energy difference during the substep
def bookkeeping_R_map(x, v, h, **kwargs):
    ''' deterministic position update '''
    x_, v_ = R_map(x, v, h)
    e_diff = potential(x_) - potential(x)
    return x_, v_, e_diff


def bookkeeping_V_map(x, v, h, **kwargs):
    ''' deterministic velocity update '''
    x_, v_ = V_map(x, v, h)
    e_diff = kinetic_energy(v_) - kinetic_energy(v)
    return x_, v_, e_diff


def bookkeeping_O_map(x, v, h, gamma, **kwargs):
    ''' stochastic velocity update'''
    x_, v_ = O_map(x, v, h, gamma)
    e_diff = kinetic_energy(v_) - kinetic_energy(v)
    return x_, v_, e_diff


bookkeeping_map = {'R': bookkeeping_R_map,
                   'V': bookkeeping_V_map,
                   'O': bookkeeping_O_map}


def bookkeeping_langevin_map(x, v, h, splitting, integrator_params=None):
    W_shad, Q = 0, 0
    for substep in splitting:
        h_substep = h / sum([l == substep for l in splitting])
        x, v, e_diff = bookkeeping_map[substep](x, v, h_substep, **integrator_params)
        if substep in 'RV':
            W_shad += e_diff
        else:
            Q += e_diff
    return x, v, W_shad, Q


def bookkeeping_langevin_factory(n_steps, splitting='VRORV'):
    def multistep_langevin(x, v, h, integrator_params):
        xs, vs = np.array([np.ones_like(x)] * n_steps), np.array([np.ones_like(v)] * n_steps)
        xs[0], xs[0] = x, v
        W_shad, Q = np.zeros(n_steps), np.zeros(n_steps)
        for i in range(1, n_steps):
            xs[i], vs[i], W_shad[i], Q[i] = \
                bookkeeping_langevin_map(xs[i - 1], vs[i - 1], h, splitting, integrator_params)
        return xs, vs, W_shad, Q

    return multistep_langevin


def test_scheme(x_0, v_0, n_steps, scheme, h, integrator_params):
    langevin = bookkeeping_langevin_factory(n_steps, scheme)
    xs, vs, W_shad, Q = langevin(x_0, v_0, h, integrator_params)

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
    plt.plot(xs[:,0], vs[:,0] * m, linewidth=0.5)
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


schemes = "VRORV RVOVR OVRVO".split()
results = dict()
print("Simulating long trajectories using {} integrators".format(schemes))
for scheme in schemes:
    results[scheme] = test_scheme(np.random.randn(dim), draw_velocities(),
                                  n_steps=int(1e4), scheme=scheme, h=timestep,
                                  integrator_params=integrator_params
                                  )

# let's confirm that energy change accumulated by the bookkeeper is the same as the
# actual energy change
print("Confirming that bookkeeping is self-consistent...")
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
              'accumulated and actual energy change: {:.3f}'.format(scheme, np.mean(np.linalg.norm(discrepancy))))

check_bookkeeping(results)


### Estimate nonequilibrium free energy difference

print("Collecting equilibrium samples...")
# collect equilibrium samples
import emcee
n_walkers = 2 * dim
sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=dim, lnpostfn=log_q)
sampler.run_mcmc(pos0=np.random.randn(n_walkers, dim), N=50000)
equilibrium_samples = sampler.flatchain[1000:]
draw_configuration = lambda : equilibrium_samples[np.random.randint(len(equilibrium_samples))]
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
    else:
        raise (ValueError("Invalid midpoint operator"))

    # simulate starting from out-of-equilibrium ensemble
    _, _, W_shad, _ = integrator(x_1, v_1)
    W_shad_R = np.cumsum(W_shad)

    return W_shad_F, W_shad_R, x_1

def perform_benchmark(integrator, midpoint_operator, n_samples):
    samples = [run_protocol(integrator, midpoint_operator) for _ in tqdm(range(n_samples))]
    W_shads_F, W_shads_R, x = [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]
    return W_shads_F, W_shads_R, x

def estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R, N_eff=None, verbose=True):
    if N_eff == None: N_eff = len(W_shads_F)
    DeltaF_neq = 0.5 * (np.mean(W_shads_F) - np.mean(W_shads_R))
    # squared uncertainty = [var(W_0->M) + var(W_M->2M) - 2 cov(W_0->M, W_M->2M)] / (4 N_eff)
    sq_uncertainty = (np.var(W_shads_F) + np.var(W_shads_R) - 2 * np.cov(W_shads_F, W_shads_R)[0, 1]) / (4 * N_eff)
    if verbose:
        print("\t<W_F>: {:.3f} +/- {:.3f}".format(np.mean(W_shads_F), 1.96 * np.std(W_shads_F) / np.sqrt(len(W_shads_F))))
        print("\t<W_R>: {:.3f} +/- {:.3f}".format(np.mean(W_shads_R), 1.96 * np.std(W_shads_R) / np.sqrt(len(W_shads_R))))
        print("\tcov(W_F, W_R): {:.3f}".format(np.cov(W_shads_F, W_shads_R)[0, 1]))
    return DeltaF_neq, sq_uncertainty


def plot_mean_and_95_confidence_bands(array, x=None, color="blue", label=None):
    """Given a list of trajectories, plot the mean trajectory +/- 1.96 * standard error"""
    mean = np.mean(array, 0)
    band = 1.96 * np.std(array, 0) / np.sqrt(len(array))

    plt.plot(mean, color=color, label=label)
    if x == None: x = range(len(array[0]))
    plt.fill_between(x, mean - band, mean + band, color=color, alpha=0.3)


def unpack_W_shads(W_shads_F, W_shads_R):
    return [F[-1] for F in W_shads_F], [R[-1] for R in W_shads_R]


colors = {"null": "blue",
          "randomize-velocity": "green"
          }


n_steps = 20
results = dict()
midpoint_operators = ["null", "randomize-velocity"]
for scheme in schemes:
    print("Scheme: {}".format(scheme))

    integrator = partial(bookkeeping_langevin_factory(n_steps, scheme), h=timestep, integrator_params=integrator_params)

    results[scheme] = dict()
    plt.figure()
    for midpoint_operator in midpoint_operators:
        print("Midpoint operator: {}".format(midpoint_operator))
        W_shads_F, W_shads_R, x = perform_benchmark(integrator, midpoint_operator, n_samples=10000)
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

        print("\tDeltaF_neq: {:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(sq_uncertainty)))

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
