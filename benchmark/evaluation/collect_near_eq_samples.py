import numpy as np
from simtk import unit

from benchmark import simulation_parameters
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import NonequilibriumSimulator
from benchmark.testsystems import water_cluster_rigid

# experiment variables
testsystems = {
    "water_cluster_rigid": water_cluster_rigid
}
splittings = {"OVRVO": "O V R V O",
              "ORVRO": "O R V R O",
              "RVOVR": "R V O V R",
              "VRORV": "V R O R V",
              }
marginals = ["configuration", "full"]
dt_range = np.array([0.1] + list(np.arange(0.5, 8.001, 0.5))) * unit.femtosecond

# constant parameters
collision_rates = {'100' : 100.0 / unit.picoseconds,
                   '10' : 10.0 / unit.picoseconds,
                   '1' : 1.0 / unit.picoseconds,
                   '0.1' : 0.1 / unit.picoseconds}
temperature = simulation_parameters['temperature']
n_steps = 1000  # number of steps until system is judged to have reached "steady-state"
n_samples = 100000

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


def save(job_id, experiment, result):
    from pickle import dump

    with open("{}.pkl".format(job_id), 'wb') as f:
        dump((experiment, result), f)


if __name__ == '__main__':
    experiments = []
    for scheme in splittings:
        for dt in dt_range:
            for marginal in marginals:
                for testsystem in testsystems:
                    for collision_rate in collision_rates:
                        experiments.append((scheme, dt, marginal, testsystem, collision_rate))

    print(len(experiments))

    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No valid job_id supplied! Selecting one at random")
        job_id = np.random.randint(len(experiments)) + 1

    experiment = experiments[job_id - 1]
    print(experiment)

    (scheme, dt, marginal, testsystem, collision_rate) = experiment
    noneq_sim = noneq_sim_factory(testsystem, scheme, dt, collision_rates[collision_rate])
    result = noneq_sim.collect_protocol_samples(n_samples, n_steps, marginal)

    save(job_id, experiment, result)
