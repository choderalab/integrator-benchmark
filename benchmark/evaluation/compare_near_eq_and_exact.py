import numpy as np
from benchmark.testsystems import waterbox_constrained, t4_constrained, alanine_constrained
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems.bookkeepers import get_state_as_mdtraj
from benchmark import simulation_parameters
from simtk import unit
from tqdm import tqdm

testsystems = {
    "waterbox_constrained": waterbox_constrained,
    "t4_constrained": t4_constrained
}

temperature = simulation_parameters['temperature']

splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }

dt_range = np.array([0.1] + list(np.arange(0.5, 4.001, 0.5)))
marginals = ["configuration", "full"]

collision_rate = 1.0 / unit.picoseconds

n_inner_samples = 100
n_outer_samples = 100
n_steps = 1000


def stdev_log_rho_pi(w):
    """Estimate the standard deviation of the estimate of log < e^{-w} >_{x; \Lambda}

    Note : This will be an underestimate esp. when len(w) is small or stdev_log_rho_pi is large.
    """
    return np.std(np.exp(-w)) / (np.mean(np.exp(-w)) * np.sqrt(len(w)))

def estimate_from_work_samples(work_samples):
    """Returns an estimate of log(rho(x) / pi(x)) from unitless work_samples initialized at x"""
    return np.log(np.mean(np.exp(-np.array(work_samples))))


def inner_sample(noneq_sim, x, v, n_steps, marginal="full"):
    if marginal == "full":
        pass
    elif marginal == "configuration":
        v = noneq_sim.sample_v_given_x(x)
    else:
        raise(Exception("marginal must be `full` or `configuration`"))

    return noneq_sim.accumulate_shadow_work(x, v, n_steps)['W_shad']

def collect_inner_samples_naive(x, v, noneq_sim, marginal="full", n_steps=1000, n_inner_samples=100):
    """Collect a fixed number of noneq trajectories starting from x,v"""
    Ws = np.zeros(n_inner_samples)
    for i in range(n_inner_samples):
        Ws[i] = inner_sample(noneq_sim, x, v, n_steps, marginal)
    return Ws

def collect_inner_samples_until_threshold(x, v, noneq_sim, marginal="full", initial_size=100, batch_size=5, n_steps=1000, threshold=0.1, max_samples=1000):
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
        raise(RuntimeWarning("stdev_log_rho_pi(Ws) > threshold\n({:.3f} > {:.3f})".format(stdev_log_rho_pi(np.array(Ws)), threshold)))

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


def outer_sample_adaptive(index=0, noneq_sim=None, marginal="full", n_steps=1000, initial_size=100, batch_size=5, threshold=0.1, max_samples=1000):
    rho_sample = sample_from_rho(noneq_sim, n_steps)
    x, v, W_shad_forward = rho_sample['x'], rho_sample['v'], rho_sample['W_shad_forward']

    Ws = collect_inner_samples_until_threshold(x, v, noneq_sim, marginal, initial_size, batch_size, n_steps, threshold, max_samples)

    return {
        "xv": (x, v),
        "Ws": Ws,
        "estimate": estimate_from_work_samples(Ws),
        "W_shad_forward": W_shad_forward
    }

def estimate_kl_div(testsystem_name, scheme, dt, marginal, collision_rate,
                    n_inner_samples=100, n_outer_samples=100, n_steps=1000, return_samples=False):

    testsystem = testsystems[testsystem_name]
    integrator = LangevinSplittingIntegrator(splittings[scheme],
                                             temperature=temperature,
                                             collision_rate=collision_rate,
                                             timestep=dt * unit.femtosecond)
    noneq_sim = NonequilibriumSimulator(testsystem, integrator)

    outer_samples = []
    for _ in tqdm(range(n_outer_samples)):
        outer_samples.append(outer_sample_naive(noneq_sim=noneq_sim, marginal=marginal, n_inner_samples=n_inner_samples, n_steps=n_steps))

    # estimate D_KL as a sample average over x ~ rho of log(rho(x) / pi(x))
    new_estimate = np.mean([s["estimate"] for s in outer_samples], 0)

    # near-equilibrium estimate
    W_F_hat = np.mean([s["W_shad_forward"] for s in outer_samples])
    W_R_hat = np.mean([s["Ws"][0] for s in outer_samples])  # picking the first W_R sample associated with each W_F
    near_eq_estimate = 0.5 * (W_F_hat - W_R_hat)


    result = {
        "new_estimate": new_estimate,
        "near_eq_estimate": near_eq_estimate,
        "W_shad_forward": np.array([s["W_shad_forward"] for s in outer_samples]),
        "Ws": np.array([s["Ws"] for s in outer_samples])
    }

    if return_samples: # takes up a lot of extra space, maybe unnecessary
        result['samples'] = outer_samples

    return result


def save(job_id, experiment, result):
    from pickle import dump

    with open( "{}.pkl".format(job_id), 'wb') as f:
        dump((experiment, result), f)

if __name__ == '__main__':
    experiments = []
    for scheme in splittings:
        for dt in dt_range:
            for marginal in marginals:
                for testsystem in testsystems:
                    experiments.append((scheme, dt, marginal, testsystem))

    print(len(experiments))

    import sys

    try:
        job_id = int(sys.argv[1])
        experiment = experiments[job_id - 1]
        print(experiment)
        (scheme, dt, marginal, testsystem) = experiment
        result = estimate_kl_div(testsystem, scheme, dt, marginal, collision_rate, n_inner_samples, n_outer_samples, n_steps)
        save(job_id, experiment, result)

    except:
        print("No job_id supplied!")
