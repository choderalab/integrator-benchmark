import os

import numpy as np
from benchmark import DATA_PATH
from benchmark.integrators import metropolis_hastings_factory
from openmmtools.testsystems import CustomExternalForcesTestSystem, ConstraintCoupledHarmonicOscillator
from tqdm import tqdm

n_particles = 500


def load_harmonic_oscillator(*args, **kwargs):
    """Load 3D harmonic oscillator"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^2 + {k}*y^2 + {k}*z^2".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions


def load_constraint_coupled_harmonic_oscillators(*args, **kwargs):
    """Load pair of constraint-coupled 3D harmonic oscillators"""
    testsystem = ConstraintCoupledHarmonicOscillator()
    return testsystem.topology, testsystem.system, testsystem.positions


def load_quartic_potential(*args, **kwargs):
    """Load 3D quartic potential"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions


def load_mts_test(*args, **kwargs):
    """
    n_particles : int
        number of identical, independent particles to add
        this is just an efficiency thing -- can simulate many replicates in parallel, instead of spending
        the openmm overhead to get a single replicate at a time

        to-do: maybe have the customintegrator keep track of the shadow work of each DOF separately?
            that way, our estimates / uncertainty estimates aren't messed up (e.g. it doesn't look like
            we have just 1,000 samples, when in fact we have 500,000 samples)
        to-do: maybe have the customintegrator keep track of the shadow work due to each force group separately?
    """
    ks = [100.0, 400.0]  # stiffness of each force group term
    # force group term 0 will be evaluated most slowly, etc...
    testsystem = CustomExternalForcesTestSystem(
        energy_expressions=["{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=k) for k in ks],
        n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions


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

    def collect_equilibrium_samples(self, n_samples=1000000):
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


quartic = NumbaBookkeepingSimulator()
double_well = NumbaBookkeepingSimulator(potential=lambda x: x ** 6 + 2 * np.cos(5 * (x + 1)),
                                        force=lambda x: - (6 * x ** 5 - 10 * np.sin(5 * (x + 1))),
                                        name='double_well'
                                        )


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
        return self.integrator(x_0, v_0, n_steps)[-1][-1]

    def collect_protocol_samples(self, n_protocol_samples, protocol_length, marginal="configuration"):
        """Perform nonequilibrium measurements, aimed at measuring the free energy difference for the chosen marginal."""
        W_shads_F, W_shads_R = [], []
        for _ in tqdm(range(n_protocol_samples)):
            x_0 = self.sample_x_from_equilibrium()
            v_0 = self.sample_v_given_x(x_0)
            xs, vs, Q, W_shads = self.integrator(x0=x_0, v0=v_0, n_steps=protocol_length)
            W_shads_F.append(W_shads[-1])

            x_1 = xs[-1][-1]
            if marginal == "configuration":
                v_1 = self.sample_v_given_x(x_1)
            elif marginal == "full":
                v_1 = vs[-1][-1]
            else:
                raise NotImplementedError("`marginal` must be either 'configuration' or 'full'")

            W_shads_R.append(self.accumulate_shadow_work(x_1, v_1, protocol_length))

        return np.array(W_shads_F), np.array(W_shads_R)
