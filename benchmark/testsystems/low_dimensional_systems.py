import numpy as np
from openmmtools.testsystems import CustomExternalForcesTestSystem, AlanineDipeptideVacuum, WaterBox, AlanineDipeptideExplicit, SrcImplicit
import os

from bookkeepers import BookkeepingSimulator
from benchmark.integrators import metropolis_hastings_factory
from benchmark import DATA_PATH

n_particles = 500
def load_harmonic_oscillator(**args):
    """Load 3D harmonic oscillator"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^2 + {k}*y^2 + {k}*z^2".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions

def load_quartic_potential(**args):
    """Load 3D quartic potential"""
    testsystem = CustomExternalForcesTestSystem(("{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=100.0),),
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions

def load_mts_test(**args):
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
    ks = [100.0, 400.0] # stiffness of each force group term
    # force group term 0 will be evaluated most slowly, etc...
    testsystem = CustomExternalForcesTestSystem(energy_expressions=["{k}*x^4 + {k}*y^4 + {k}*z^4".format(k=k) for k in ks],
                                                n_particles=n_particles)
    return testsystem.topology, testsystem.system, testsystem.positions

class NumbaBookkeepingQuarticSimulator(BookkeepingSimulator):
    def __init__(self, mass=10.0, beta=1.0):
        self.beta = beta
        self.velocity_scale =np.sqrt(1.0 / (beta * mass))
        sigma2 = self.velocity_scale ** 2

        self.q = lambda x : np.exp(-x**4)
        # timestep = 1.0
        # gamma = 100.0

        def potential(x):
            return x ** 4

        def force(x):
            return - 4.0 * x ** 3

        def reduced_potential(x):
            return potential(x) * beta

        def log_q(x):
            return - reduced_potential(x)

        def q(x):
            return np.exp(log_q(x))

        self.equilibrium_simulator = metropolis_hastings_factory(q)
        self.name = "quartic"
        self._path_to_samples = self.get_path_to_samples()

        # Load or simulate
        self.load_or_simulate_x_samples()

    def load_or_simulate_x_samples(self):
        """If we've already collected and stored equilibrium samples, load those
        Otherwise, collect equilibrium samples"""
        self._path_to_samples = self.get_path_to_samples()
        if self.check_for_cached_samples():
            self.x_samples = self.load_equilibrium_samples()
        else:
            self.x_samples = self.collect_equilibrium_samples()
            self.save_equilibrium_samples(self.x_samples)

    def collect_equilibrium_samples(self, n_samples=10000):
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
        return self.x_samples[np.random.randint(len(self.x_samples))]

    def sample_v_from_equilibrium(self):
        """Sample velocity marginal."""
        return np.random.randn() * self.velocity_scale

    def accumulate_shadow_work(self, x_0, v_0, n_steps):
        """Simulate for n_steps, starting at x_0, v_0.
        Returns a length n_steps numpy array of shadow_work values"""


