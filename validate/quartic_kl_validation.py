import os
from collections import namedtuple
from functools import partial
from pickle import dump

import numpy as np
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from numba import jit

DATA_PATH = "quartic_results"

ExperimentDescriptor = namedtuple("ExperimentDescriptor",
                                  ["experiment_name", "system_name", "equilibrium_simulator",
                                   "splitting", "integrator", "timestep", "marginal", "gamma",
                                   "n_protocol_samples", "protocol_length"])


class Experiment():
    def __init__(self, experiment_descriptor, filename):
        self.experiment_descriptor = experiment_descriptor
        self.filename = filename

    def run(self):
        exp = self.experiment_descriptor
        timestep = exp.timestep
        gamma = exp.gamma
        integrator = exp.integrator
        simulator = NumbaNonequilibriumSimulator(exp.equilibrium_simulator,
                                                 partial(integrator, gamma=gamma, dt=timestep))

        if exp.marginal == "configuration":
            sample_collector = simulator.collect_conf_protocol_samples
        else:
            sample_collector = simulator.collect_full_protocol_samples

        W_shads_F, W_shads_R = sample_collector(exp.n_protocol_samples, exp.protocol_length)

        self.result = {}
        self.result["W_shads_F"] = W_shads_F
        self.result["W_shads_R"] = W_shads_R

        DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R)
        print(self)
        print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    def save(self):
        with open(self.filename, "wb") as f:
            everything_but_the_simulator = self.experiment_descriptor._asdict()
            everything_but_the_simulator.pop("equilibrium_simulator")

            dump({"result": self.result, "descriptor": everything_but_the_simulator}, f)

    def run_and_save(self):
        self.run()
        self.save()

    def __str__(self):
        exp = self.experiment_descriptor

        properties = [exp.system_name,
                      exp.splitting,
                      "dt={}fs".format(exp.timestep),
                      "marginal: {}".format(exp.marginal),
                      "gamma: {}".format(exp.gamma),
                      ]

        return "\n\t".join(["{}"] * len(properties)).format(*properties)


def jit_if_possible(f):
    """If the function isn't already
    JIT-compiled, JIT it now."""
    try:
        f = jit(f)
    except:
        pass
    return f


# these integrators return three numbers: x,v,W_shad
def ovrvo_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_ovrvo(x0, v0, n_steps, gamma, dt):
        x, v = x0, v0
        W_shad = 0

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            E_old = 0.5 * m * v ** 2 + potential(x)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + (dt * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            E_new = 0.5 * m * v ** 2 + potential(x)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            # Update W_shads
            W_shad += (E_new - E_old)

        return x, v, W_shad

    return jit(simulate_ovrvo)


def orvro_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_orvro(x0, v0, n_steps, gamma, dt):
        x, v = x0, v0
        W_shad = 0

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            E_old = 0.5 * m * v ** 2 + potential(x)

            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)

            E_new = 0.5 * m * v ** 2 + potential(x)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            # Update W_shads
            W_shad += (E_new - E_old)

        return x, v, W_shad

    return jit(simulate_orvro)


def vrorv_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_vrorv(x0, v0, n_steps, gamma, dt):
        x, v = x0, v0
        W_shad = 0

        a = np.exp(-gamma * (dt / 1.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 1.0)))

        for i in range(1, n_steps):
            E_old = 0.5 * m * v ** 2 + potential(x)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)
            E_new = 0.5 * m * v ** 2 + potential(x)

            W_shad += (E_new - E_old)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            E_old = 0.5 * m * v ** 2 + potential(x)
            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            E_new = 0.5 * m * v ** 2 + potential(x)

            W_shad += (E_new - E_old)

        return x, v, W_shad

    return jit(simulate_vrorv)


def rvovr_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_rvovr(x0, v0, n_steps, gamma, dt):
        x, v = x0, v0
        W_shad = 0

        a = np.exp(-gamma * (dt / 1.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 1.0)))

        for i in range(1, n_steps):
            E_old = 0.5 * m * v ** 2 + potential(x)
            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            E_new = 0.5 * m * v ** 2 + potential(x)

            W_shad += (E_new - E_old)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            E_old = 0.5 * m * v ** 2 + potential(x)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)
            E_new = 0.5 * m * v ** 2 + potential(x)

            W_shad += (E_new - E_old)

        return x, v, W_shad

    return jit(simulate_rvovr)


def metropolis_hastings_factory(q):
    q = jit_if_possible(q)

    def rw_metropolis_hastings(x0, n_steps):
        xs = np.zeros(n_steps)
        xs[0] = x0

        # draw all the random numbers we'll need
        proposal_eps = np.random.randn(n_steps)  # standard normal
        accept_eps = np.random.rand(n_steps)  # uniform(0,1)

        for i in range(1, n_steps):
            x_prop = xs[i - 1] + proposal_eps[i]
            a_r_ratio = q(x_prop) / q(xs[i - 1])

            # accept / reject
            if a_r_ratio > accept_eps[i]:
                xs[i] = x_prop
            else:
                xs[i] = xs[i - 1]
        return xs

    return jit(rw_metropolis_hastings)


class NumbaBookkeepingSimulator():
    def __init__(self, mass=10.0, beta=1.0,
                 potential=lambda x: x ** 4,
                 force=lambda x: -4.0 * x ** 3,
                 name='quartic'
                 ):
        self.mass = mass
        self.beta = beta
        self.velocity_scale = np.sqrt(1.0 / (beta * mass))

        def reduced_potential(x):
            return potential(x) * beta

        def log_q(x):
            return - reduced_potential(x)

        def q(x):
            return np.exp(log_q(x))

        self.potential = potential
        self.force = force
        self.reduced_potential = reduced_potential
        self.log_q = log_q
        self.q = q
        self.equilibrium_simulator = metropolis_hastings_factory(q)
        self.name = name
        self._path_to_samples = self.get_path_to_samples()
        self.cached = False

    def load_or_simulate_x_samples(self):
        """If we've already collected and stored equilibrium samples, load those
        Otherwise, collect equilibrium samples"""

        self._path_to_samples = self.get_path_to_samples()
        if self.check_for_cached_samples():
            print("Cache found: loading...")
            self.x_samples = self.load_equilibrium_samples()
        else:
            print("Cache not found: collecting equilibrium samples...")
            self.x_samples = self.collect_equilibrium_samples()
            self.save_equilibrium_samples(self.x_samples)
        self.cached = True

    def collect_equilibrium_samples(self, n_samples=10000000):
        """Collect equilibrium samples, return as (n_samples, ) numpy array"""
        equilibrium_samples = self.equilibrium_simulator(x0=np.random.randn(), n_steps=n_samples)
        return np.array(equilibrium_samples)

    def get_path_to_samples(self):
        """Samples are {name}_samples.npy in DATA_PATH"""
        return os.path.join(DATA_PATH, '{}_samples.npy'.format(self.name))

    def check_for_cached_samples(self):
        """Check if there's a file where we expect to find cached
        equilibrium samples.
        """
        # TODO : Need to check if any of the simulation parameters have changed.
        return os.path.exists(self._path_to_samples)

    def save_equilibrium_samples(self, x_samples):
        """Save numpy archive of equilibrium samples to disk."""
        np.save(self._path_to_samples, x_samples)

    def load_equilibrium_samples(self):
        """Load numpy archive of equilibrium samples"""
        print("Loading equilibrium samples from {}...".format(self._path_to_samples))
        x_samples = np.load(self._path_to_samples)
        return x_samples

    def sample_x_from_equilibrium(self):
        """Draw sample (uniformly, with replacement) from cache of configuration samples"""
        if self.cached == False:
            self.load_or_simulate_x_samples()
        return self.x_samples[np.random.randint(len(self.x_samples))]

    def sample_v_given_x(self, x):
        """Sample velocity marginal. (Here, p(v) = p(v|x).)"""
        return np.random.randn() * self.velocity_scale


class NumbaNonequilibriumSimulator():
    """Nonequilibrium simulator, supporting shadow_work accumulation, and drawing x, v, from equilibrium.

    Numba integrators do this: xs, vs, Q, W_shads = numba_integrator(x0, v0, n_steps)
    """

    def __init__(self, equilibrium_simulator, integrator):
        self.equilibrium_simulator, self.integrator = equilibrium_simulator, integrator

    def sample_x_from_equilibrium(self):
        """Draw sample (uniformly, with replacement) from cache of configuration samples"""
        return self.equilibrium_simulator.sample_x_from_equilibrium()

    def sample_v_given_x(self, x):
        """Sample velocities from Maxwell-Boltzmann distribution."""
        return self.equilibrium_simulator.sample_v_given_x(x)

    def accumulate_shadow_work(self, x_0, v_0, n_steps):
        """Run the integrator for n_steps and return the shadow work accumulated"""
        return self.integrator(x_0, v_0, n_steps)[-1]

    @jit
    def collect_conf_protocol_samples(self, n_protocol_samples, protocol_length):
        """Perform nonequilibrium measurements, aimed at measuring the free energy difference in the x marginal"""
        W_shads_F, W_shads_R = np.zeros(n_protocol_samples), np.zeros(n_protocol_samples)

        v_scale = self.equilibrium_simulator.velocity_scale

        for i in range(n_protocol_samples):
            x = self.equilibrium_simulator.x_samples[np.random.randint(len(self.equilibrium_simulator.x_samples))]
            v = np.random.randn() * v_scale
            x, v, W = self.integrator(x0=x, v0=v, n_steps=protocol_length)

            W_shads_F[i] = W

            v = np.random.randn() * v_scale
            x, v, W = self.integrator(x0=x, v0=v, n_steps=protocol_length)

            W_shads_R[i] = W

        return W_shads_F, W_shads_R

    @jit
    def collect_full_protocol_samples(self, n_protocol_samples, protocol_length):
        """Perform nonequilibrium measurements, aimed at measuring the free energy difference."""
        W_shads_F, W_shads_R = np.zeros(n_protocol_samples), np.zeros(n_protocol_samples)

        v_scale = self.equilibrium_simulator.velocity_scale

        for i in range(n_protocol_samples):
            x = self.equilibrium_simulator.x_samples[np.random.randint(len(self.equilibrium_simulator.x_samples))]
            v = np.random.randn() * v_scale
            x, v, W = self.integrator(x0=x, v0=v, n_steps=protocol_length)

            W_shads_F[i] = W

            x, v, W = self.integrator(x0=x, v0=v, n_steps=protocol_length)

            W_shads_R[i] = W

        return W_shads_F, W_shads_R


n_protocol_samples, protocol_length = 5000000, 100
system_name = "quartic"
equilibrium_simulator = NumbaBookkeepingSimulator()
potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
schemes = {"VRORV": vrorv_factory(potential, force, velocity_scale, mass),
           "RVOVR": rvovr_factory(potential, force, velocity_scale, mass),
           "ORVRO": orvro_factory(potential, force, velocity_scale, mass),
           "OVRVO": ovrvo_factory(potential, force, velocity_scale, mass),
           }
dt_range = np.linspace(0.1, 1.1, 10)[::-1]
gamma = 100000.0

experiment_name = "0_quartic_validation"
experiments = []
i = 0
for splitting in sorted(schemes.keys()):
    for dt in dt_range:
        for marginal in ["configuration", "full"]:
            partial_fname = "{}_{}.pkl".format(experiment_name, i)
            full_filename = os.path.join("quartic_results", partial_fname)

            experiment_descriptor = ExperimentDescriptor(
                experiment_name=experiment_name,
                system_name=system_name,
                equilibrium_simulator=equilibrium_simulator,
                splitting=splitting,
                integrator=schemes[splitting],
                timestep=dt,
                marginal=marginal,
                gamma=gamma,
                n_protocol_samples=n_protocol_samples,
                protocol_length=protocol_length,
            )

            experiments.append(Experiment(experiment_descriptor, full_filename))
            i += 1

if __name__ == "__main__":
    _ = equilibrium_simulator.sample_x_from_equilibrium()

    from time import time

    t0 = time()

    import sys

    do_experiment = False

    try:
        job_id = int(sys.argv[1]) - 1
        do_experiment = True
        np.random.seed(job_id)
    except:
        print("no input received!")

    if do_experiment:
        experiments[job_id].run_and_save()
        t1 = time()
        print("elapsed time: {:.4f}s".format(t1 - t0))
