import warnings

import numpy as np
from simtk import unit
from tqdm import tqdm

from benchmark import simulation_parameters
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems import water_cluster_rigid, alanine_constrained
from benchmark.testsystems.bookkeepers import get_state_as_mdtraj

# experiment variables
testsystems = {
    "alanine_constrained": alanine_constrained,
    "water_cluster_rigid": water_cluster_rigid,
}
splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }
marginals = ["configuration", "full"]
dt_range = np.array([0.1] + list(np.arange(0.5, 8.001, 0.5))) * unit.femtosecond

# constant parameters
collision_rate = 1.0 / unit.picoseconds
temperature = simulation_parameters['temperature']

def n_steps_(dt, n_collisions=1, max_steps=1000):
    """Heuristic for how many steps are needed to reach steady state:
    should run at least long enough to have n_collisions full "collisions"
    with the bath.

    This corresponds to more discrete steps when dt is small, and fewer discrete steps
    when dt is large.

    Examples:
        n_steps_(dt=1fs) = 1000
        n_steps_(dt=2fs) = 500
        n_steps_(dt=4fs) = 250
        n_steps_(dt=8fs) = 125
    """
    return min(max_steps, int((n_collisions / collision_rate) / dt))


# adaptive inner-loop params
inner_loop_initial_size = 50
inner_loop_batch_size = 1
inner_loop_stdev_threshold = 0.01
inner_loop_max_samples = 50000

# adaptive outer-loop params
outer_loop_initial_size = 50
outer_loop_batch_size = 1
outer_loop_stdev_threshold = inner_loop_stdev_threshold
outer_loop_max_samples = 1000


def stdev_log_rho_pi(w):
    """Approximate the standard deviation of the estimate of log < e^{-w} >_{x; \Lambda}

    Parameters
    ----------
    w : unitless (kT) numpy array of work samples

    Returns
    -------
    stdev : float

    Notes
    -----
    This will be an underestimate esp. when len(w) is small or stdev_log_rho_pi is large.
    """

    assert(type(w) != unit.Quantity) # assert w is unitless
    assert(type(w) == np.ndarray) # assert w is a numpy array

    # use leading term in taylor expansion: anecdotally, looks like it's in good agreement with
    # bootstrapped uncertainty estimates up to ~0.5-0.75, then becomes an increasingly bad underestimate
    return np.std(np.exp(-w)) / (np.mean(np.exp(-w)) * np.sqrt(len(w)))


def stdev_kl_div(outer_samples):
    """Approximate the stdev of the estimate of E_rho log_rho_pi"""

    # TODO: Propagate uncertainty from the estimates of log_rho_pi
    # currently, just use standard error of mean of log_rho_pi_s
    log_rho_pi_s = np.array([np.log(np.mean(np.exp(-sample['Ws']))) for sample in outer_samples])
    return np.std(log_rho_pi_s) / np.sqrt(len(log_rho_pi_s))


def estimate_from_work_samples(work_samples):
    """Returns an estimate of log(rho(x) / pi(x)) from unitless work_samples initialized at x"""
    return np.log(np.mean(np.exp(-np.array(work_samples))))


def inner_sample(noneq_sim, x, v, n_steps, marginal="full"):
    if marginal == "full":
        pass
    elif marginal == "configuration":
        v = noneq_sim.sample_v_given_x(x)
    else:
        raise (Exception("marginal must be `full` or `configuration`"))

    return noneq_sim.accumulate_shadow_work(x, v, n_steps)['W_shad']


def collect_inner_samples_naive(x, v, noneq_sim, marginal="full", n_steps=1000, n_inner_samples=100):
    """Collect a fixed number of noneq trajectories starting from x,v"""
    Ws = np.zeros(n_inner_samples)
    for i in range(n_inner_samples):
        Ws[i] = inner_sample(noneq_sim, x, v, n_steps, marginal)
    return Ws


def collect_inner_samples_until_threshold(x, v, noneq_sim, marginal="full", initial_size=100, batch_size=5,
                                          n_steps=1000, threshold=0.1, max_samples=1000):
    """Collect up to max_samples trajectories, potentially fewer if stdev of estimated log(rho(x,v) / pi(x,v)) is below threshold."""
    Ws = []

    # collect initial samples
    for _ in range(initial_size):
        Ws.append(inner_sample(noneq_sim, x, v, n_steps, marginal))

    # keep adding batches until either stdev threshold is reached or budget is reached
    while (stdev_log_rho_pi(np.array(Ws)) > threshold) and (len(Ws) <= (max_samples - batch_size)):
        for _ in range(batch_size):
            Ws.append(inner_sample(noneq_sim, x, v, n_steps, marginal))

    # warn user if stdev threshold was not met
    if (stdev_log_rho_pi(np.array(Ws)) > threshold):
        message = "stdev_log_rho_pi(Ws) > threshold\n({:.3f} > {:.3f})".format(stdev_log_rho_pi(np.array(Ws)), threshold)
        warnings.warn(message, RuntimeWarning)


    return np.array(Ws)


def sample_from_rho(noneq_sim, n_steps=1000):
    # (x0, v0) drawn from pi
    x0 = noneq_sim.sample_x_from_equilibrium()
    v0 = noneq_sim.sample_v_given_x(x0)

    # (x, v) drawn from rho
    W_shad_forward = noneq_sim.accumulate_shadow_work(x0, v0, n_steps)["W_shad"]
    x = get_state_as_mdtraj(noneq_sim.simulation)
    v = noneq_sim.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    return {'x': x,
            'v': v,
            'W_shad_forward': W_shad_forward,
            }


def outer_sample_naive(index=0, noneq_sim=None, marginal="full", n_inner_samples=100, n_steps=1000):
    rho_sample = sample_from_rho(noneq_sim, n_steps)
    x, v, W_shad_forward = rho_sample['x'], rho_sample['v'], rho_sample['W_shad_forward']

    Ws = collect_inner_samples_naive(x, v, noneq_sim, marginal, n_steps, n_inner_samples)

    return {
        "xv": (x, v),
        "Ws": Ws,
        "estimate": estimate_from_work_samples(Ws),
        "W_shad_forward": W_shad_forward
    }


def outer_sample_adaptive(noneq_sim=None, marginal="full", n_steps=1000, initial_size=100, batch_size=5, threshold=0.1,
                          max_samples=1000):
    rho_sample = sample_from_rho(noneq_sim, n_steps)
    x, v, W_shad_forward = rho_sample['x'], rho_sample['v'], rho_sample['W_shad_forward']

    Ws = collect_inner_samples_until_threshold(x, v, noneq_sim, marginal, initial_size, batch_size, n_steps, threshold,
                                               max_samples)

    return {
        "xv": (x, v),
        "Ws": Ws,
        "estimate": estimate_from_work_samples(Ws),
        "W_shad_forward": W_shad_forward
    }


def noneq_sim_factory(testsystem_name, scheme, dt, collision_rate):
    """Generate a NonequilibriumSimulator object for a given experiment

    Parameters
    ----------
    testsystem_name : string
    scheme : string
    dt : in units compatible with unit.femtosecond
    collision_rate : in units compatible with (1 / unit.picosecond)

    Returns
    -------
    noneq_sim : NonequilibriumSimulator

    """
    # check that testsystem_name is valid
    assert (testsystem_name in testsystems)

    # check that scheme is valid
    assert (scheme in splittings)

    # check that dt is valid
    assert (type(dt) == unit.Quantity)
    assert (dt.unit.is_compatible(unit.femtosecond))

    # check that collision_rate is valid
    assert (type(collision_rate) == unit.Quantity)
    assert ((1 / collision_rate).unit.is_compatible(unit.picosecond))

    testsystem = testsystems[testsystem_name]
    integrator = LangevinSplittingIntegrator(splittings[scheme],
                                             temperature=temperature,
                                             collision_rate=collision_rate,
                                             timestep=dt)
    noneq_sim = NonequilibriumSimulator(testsystem, integrator)
    return noneq_sim


def process_outer_samples(outer_samples):
    """Compute "exact" estimate of D_KL from work samples"""

    # estimate D_KL as a sample average over x ~ rho of log(rho(x) / pi(x))
    new_estimate = np.mean([s["estimate"] for s in outer_samples], 0)

    return new_estimate


def estimate_kl_div_naive_outer_loop(noneq_sim, marginal,
                                     outer_sample_fxn, n_outer_samples=100, n_steps=1000,
                                     return_samples=False):
    """Collect n_outer_samples samples un-adaptively"""
    outer_samples = []
    for _ in tqdm(range(n_outer_samples)):
        outer_samples.append(outer_sample_fxn(noneq_sim=noneq_sim, marginal=marginal, n_steps=n_steps))

    new_estimate = process_outer_samples(outer_samples)

    result = {
        "new_estimate": new_estimate,
        "W_shad_forward": np.array([s["W_shad_forward"] for s in outer_samples]),
        "Ws": np.array([s["Ws"] for s in outer_samples])
    }

    if return_samples:  # takes up a lot of extra space, maybe unnecessary
        result['samples'] = outer_samples

    return result


def estimate_kl_div_adaptive_outer_loop(
        noneq_sim, marginal, outer_sample_fxn, n_steps=1000,
        initial_size=100, batch_size=1, threshold=0.1, max_samples=1000,
        return_samples=False):
    """Collect up to max_samples outer-loop samples, using a threshold on the stdev of the estimated KL divergence"""

    collect_outer_sample = lambda: outer_sample_fxn(noneq_sim=noneq_sim, marginal=marginal, n_steps=n_steps)

    outer_samples = []
    for _ in tqdm(range(initial_size)):
        outer_samples.append(collect_outer_sample())

    # keep adding batches until either stdev threshold is reached or budget is reached
    while (stdev_kl_div(outer_samples) > threshold) and (len(outer_samples) <= (max_samples - batch_size)):
        for _ in range(batch_size):
            outer_samples.append(collect_outer_sample())

    # warn user if stdev threshold was not met
    if (stdev_kl_div(outer_samples) > threshold):
        message = "stdev_kl_div(outer_samples) > threshold\n({:.3f} > {:.3f})".format(stdev_kl_div(outer_samples),
                                                                                threshold)
        warnings.warn(message, RuntimeWarning)

    new_estimate = process_outer_samples(outer_samples)

    result = {
        "new_estimate": new_estimate,
        "W_shad_forward": np.array([s["W_shad_forward"] for s in outer_samples]),
        "Ws": np.array([s["Ws"] for s in outer_samples])
    }

    if return_samples:  # takes up a lot of extra space, maybe unnecessary
        result['samples'] = outer_samples

    return result


def resample_Ws(Ws):
    """Generate a bootstrap sample of the Ws structure.

    Parameters
    ----------
    Ws : iterable of iterables
        Each element of Ws is an iterable of floats, potentially of varying length
        * len(Ws) is the number of outer-loop samples
        * [len(w) for w in Ws] are the numbers of inner-loop samples
            associated with each outer-loop sample

    Algorithm
    ---------
    * Resample elements of Ws uniformly with replacement, assign to Ws_
    * For each element in Ws_, resample its elements uniformly with replacement

    Returns
    -------
    Ws_ : iterable of iterables, same shapes as input Ws
    """
    n_outer_samples = len(Ws)

    Ws_ = Ws[np.random.randint(0, n_outer_samples, n_outer_samples)]  # resample the rows

    for i in range(n_outer_samples):
        n_cols = len(Ws_[i])
        Ws_[i] = Ws_[i][np.random.randint(0, n_cols, n_cols)]  # within each row, resample the columns independently
    return Ws_

def save(job_id, experiment, result):
    from pickle import dump

    with open("{}.pkl".format(job_id), 'wb') as f:
        dump((experiment, result), f)


from functools import partial

if __name__ == '__main__':
    experiments = []
    for scheme in splittings:
        for dt in dt_range:
            for marginal in marginals:
                for testsystem in testsystems:
                    experiments.append((scheme, dt, marginal, testsystem))

    print(len(experiments))

    outer_sample_fxn = partial(outer_sample_adaptive,
                               initial_size=inner_loop_initial_size, batch_size=inner_loop_batch_size,
                               threshold=inner_loop_stdev_threshold, max_samples=inner_loop_max_samples)

    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No valid job_id supplied! Selecting one at random")
        job_id = np.random.randint(len(experiments)) + 1

    experiment = experiments[job_id - 1]
    print(experiment)

    (scheme, dt, marginal, testsystem) = experiment
    noneq_sim = noneq_sim_factory(testsystem, scheme, dt, collision_rate)
    n_steps = n_steps_(dt)
    result = estimate_kl_div_adaptive_outer_loop(noneq_sim, marginal, outer_sample_fxn, n_steps,
                                                 initial_size=outer_loop_initial_size,
                                                 batch_size=outer_loop_batch_size,
                                                 max_samples=outer_loop_max_samples,
                                                 threshold=outer_loop_stdev_threshold)
    save(job_id, experiment, result)
