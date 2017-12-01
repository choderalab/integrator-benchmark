from openmmtools.constants import kB
import numpy as np
from benchmark.testsystems import alanine_constrained
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems.bookkeepers import get_state_as_mdtraj
from benchmark import simulation_parameters
from simtk import unit
from tqdm import tqdm


testsystem = alanine_constrained

temperature = simulation_parameters['temperature']

kT = kB * temperature
conversion_factor = unit.kilojoule_per_mole / kT

splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }


dt_range = np.array([0.1] + list(np.arange(0.5, 4.001, 0.5)))
marginals = ["configuration", "full"]

collision_rates = {"low": 1.0 / unit.picoseconds,
                   "high": 91.0 / unit.picoseconds}

n_inner_samples = 100
n_outer_samples = 1000
n_steps = 1000



def estimate_from_work_traces(work_traces):
    """Returns an estimate of log(rho(x) / pi(x)) from work_traces initialized at x"""
    return np.log(np.mean(np.exp(-np.array(work_traces)), 0))


def inner_sample(noneq_sim, x, v, n_steps, marginal="full"):

    if marginal == "full":
        pass
    elif marginal == "configuration":
        v = noneq_sim.sample_v_given_x(x)

    return noneq_sim.accumulate_shadow_work(x, v, n_steps, store_W_shad_trace=True)['W_shad_trace']


def outer_sample(index=0, noneq_sim=None, marginal="full", n_inner_samples=100, n_steps=1000):
    # (x0, v0) drawn from pi
    x0 = noneq_sim.sample_x_from_equilibrium()
    v0 = noneq_sim.sample_v_given_x(x0)

    # (x, v) drawn from rho
    W_shad_forward = noneq_sim.accumulate_shadow_work(x0, v0, n_steps)["W_shad"]
    x = get_state_as_mdtraj(noneq_sim.simulation)
    v = noneq_sim.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    Ws = []
    for _ in range(n_inner_samples):
        Ws.append(inner_sample(noneq_sim, x, v, n_steps, marginal))

    return {
        "xv": (x, v),
        "Ws": np.array(Ws),
        "estimate": estimate_from_work_traces(Ws),
        "W_shad_forward": W_shad_forward
    }


from multiprocessing import Pool
from functools import partial


def estimate_kl_div(scheme, dt, marginal, collision_rate,
                    n_inner_samples=100, n_outer_samples=100, n_steps=1000, parallel=False):

    integrator = LangevinSplittingIntegrator(splittings[scheme],
                                             temperature=temperature,
                                             collision_rate=collision_rates[collision_rate],
                                             timestep=dt * unit.femtosecond)
    noneq_sim = NonequilibriumSimulator(testsystem, integrator)

    if parallel:
        pool = Pool(8)

        collect_a_sample = partial(outer_sample, noneq_sim=noneq_sim, marginal=marginal,
                                   n_inner_samples=n_inner_samples, n_steps=n_steps)

        outer_samples = pool.map(collect_a_sample, range(n_outer_samples))

        del (pool)

    else:
        outer_samples = []
        for _ in tqdm(range(n_outer_samples)):
            outer_samples.append(outer_sample(noneq_sim=noneq_sim, marginal=marginal, n_inner_samples=n_inner_samples, n_steps=n_steps))

    # estimate D_KL as a sample average over x ~ rho of log(rho(x) / pi(x))
    new_estimate = np.mean([s["estimate"] for s in outer_samples], 0)

    # near-equilibrium estimate
    W_F_hat = np.mean([s["W_shad_forward"] for s in outer_samples])

    # TODO: I've collected many W_R samples per W_F sample: Decide whether to perform an average over these.
    W_R_hat = np.mean([s["Ws"][0, -1] for s in outer_samples])  # picking the first W_R sample associated with each W_F
    # W_R_hat = np.mean([np.mean(s["Ws"][:,-1]) for s in outer_samples]) # taking a mean over all W_R samples associated with each W_F
    near_eq_estimate = 0.5 * (W_F_hat - W_R_hat)

    return {
        "samples": outer_samples,
        "new_estimate": new_estimate,
        "near_eq_estimate": near_eq_estimate,
    }



def save(job_id, experiment, result):
    from pickle import dump

    with open( "{}.pkl".format(job_id), 'w') as f:
        dump((experiment, result), f)

if __name__ == '__main__':
    experiments = []
    for scheme in splittings:
        for dt in dt_range:
            for marginal in marginals:
                for collision_rate in collision_rates:
                    experiments.append((scheme, dt, marginal, collision_rate))

    print(len(experiments))

    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No job_id supplied! Doing job_id 1")
        job_id = 1

    experiment = experiments[job_id - 1]
    (scheme, dt, marginal, collision_rate) = experiment
    result = estimate_kl_div(scheme, dt, marginal, collision_rate, n_inner_samples, n_outer_samples, n_steps)
    save(job_id, experiment, result)
