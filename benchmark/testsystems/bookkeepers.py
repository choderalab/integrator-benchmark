import gc
import os

import numpy as np
from openmmtools.integrators import GHMCIntegrator, GradientDescentMinimizationIntegrator, VVVRIntegrator
from simtk import unit
from simtk.openmm import app
from tqdm import tqdm

from benchmark.utilities import strip_unit, get_total_energy, get_velocities, get_positions, \
    set_positions, set_velocities, remove_barostat, remove_center_of_mass_motion_remover

W_unit = unit.kilojoule_per_mole

from benchmark import DATA_PATH


class BookkeepingSimulator():
    """Abstracts away details of storage and simulation, and supports
    the following operations:
    * Sampling x from equilibrium
    * Sampling v from equilibrium
    * Accumulating shadow work over a trajectory with given initial conditions
    """

    def __init__(self):
        pass

    def sample_x_from_equilibrium(self):
        """Sample configuration marginal."""
        pass

    def sample_v_given_x(self, x):
        """Sample velocities conditioned on x."""
        pass

    def accumulate_shadow_work(self, x_0, v_0, n_steps):
        """Simulate for n_steps, starting at x_0, v_0.
        Returns a length n_steps numpy array of shadow_work values"""
        pass


class EquilibriumSimulator():
    """Simulates a system at equilibrium."""

    def __init__(self, platform, topology, system, positions, temperature, ghmc_timestep,
                 burn_in_length, n_samples, thinning_interval, name):

        self.platform = platform
        self.topology = topology
        self.system = system
        self.positions = positions
        self.temperature = temperature
        self.ghmc_timestep = ghmc_timestep
        self.burn_in_length = burn_in_length
        self.n_samples = n_samples
        self.thinning_interval = thinning_interval
        self.name = name
        self.cached = False

        # get constraint tolerance
        ghmc = GHMCIntegrator(temperature=self.temperature, timestep=self.ghmc_timestep)
        self.constraint_tolerance = ghmc.getConstraintTolerance()
        del (ghmc)
        gc.collect()

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

    def get_ghmc_acceptance_rate(self):
        """Return the number of acceptances divided by the number of proposals."""
        ghmc = self.unbiased_simulation.integrator
        n_accept = ghmc.getGlobalVariableByName("naccept")
        n_trials = ghmc.getGlobalVariableByName("ntrials")
        return float(n_accept) / n_trials

    def reset_ghmc_statistics(self):
        """Reset the number of acceptances and number of proposals."""
        ghmc = self.unbiased_simulation.integrator
        ghmc.setGlobalVariableByName("naccept", 0)
        ghmc.setGlobalVariableByName("ntrials", 0)

    def collect_equilibrium_samples(self):
        """Collect equilibrium samples, return as (n_samples, n_atoms, 3) numpy array"""
        print("Collecting equilibrium samples for '%s'..." % self.name)
        # Minimize energy by gradient descent
        print("Minimizing...")

        min_sim = self.construct_simulation(GradientDescentMinimizationIntegrator())
        min_sim.context.setPositions(self.positions)
        min_sim.context.setVelocitiesToTemperature(self.temperature)
        for _ in tqdm(range(100)):
            min_sim.step(1)
        pos = get_positions(min_sim)
        del (min_sim)
        gc.collect()

        # "Equilibrate" / "burn-in"
        # Running a bit of Langevin first improves GHMC acceptance rates?
        print("Intializing with Langevin dynamics...")
        langevin_sim = self.construct_simulation(
            VVVRIntegrator(temperature=self.temperature, timestep=self.ghmc_timestep))
        set_positions(langevin_sim, pos)
        for _ in tqdm(range(self.burn_in_length)):
            langevin_sim.step(1)
        pos = get_positions(langevin_sim)
        del (langevin_sim)
        gc.collect()

        print('"Burning in" unbiased GHMC sampler for {:.3}ps...'.format(
            (self.burn_in_length * self.ghmc_timestep).value_in_unit(unit.picoseconds)))
        self.unbiased_simulation = self.construct_simulation(
            GHMCIntegrator(temperature=self.temperature, timestep=self.ghmc_timestep))
        set_positions(self.unbiased_simulation, pos)
        for _ in tqdm(range(self.burn_in_length)):
            self.unbiased_simulation.step(1)
        print("Burn-in GHMC acceptance rate: {:.3f}%".format(100 * self.get_ghmc_acceptance_rate()))
        self.reset_ghmc_statistics()

        # Collect equilibrium samples
        print("Collecting equilibrium samples...")
        equilibrium_samples = []
        for _ in tqdm(range(self.n_samples)):
            self.unbiased_simulation.step(self.thinning_interval)
            x = self.unbiased_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
            equilibrium_samples.append(strip_unit(x))
        print("Equilibrated GHMC acceptance rate: {:.3f}%".format(100 * self.get_ghmc_acceptance_rate()))

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
        """Sample velocities from (constrained) Maxwell-Boltzmann distribution."""
        self.unbiased_simulation.context.setPositions(x)
        self.unbiased_simulation.context.setVelocitiesToTemperature(self.temperature)
        self.unbiased_simulation.context.applyVelocityConstraints(self.tolerance)
        return get_velocities(self.unbiased_simulation)

    def construct_simulation(self, integrator):
        """Construct a simulation instance given an integrator."""
        gc.collect()  # make sure that any recently deleted Contexts actually get deleted...
        simulation = app.Simulation(self.topology, self.system, integrator, self.platform)
        simulation.context.setPositions(self.positions)
        simulation.context.setVelocitiesToTemperature(self.temperature)
        return simulation


class NonequilibriumSimulator(BookkeepingSimulator):
    """Nonequilibrium simulator, supporting shadow_work accumulation, and drawing x, v, from equilibrium."""

    def __init__(self, equilibrium_simulator, integrator):
        self.equilibrium_simulator, self.integrator = equilibrium_simulator, integrator
        self.simulation = self.construct_simulation(integrator)
        self.constraint_tolerance = self.integrator.getConstraintTolerance()

    def construct_simulation(self, integrator):
        """Drop barostat and center-of-mass motion remover, then construct_simulation"""
        remove_barostat(self.equilibrium_simulator.system)
        remove_center_of_mass_motion_remover(self.equilibrium_simulator.system)
        return self.equilibrium_simulator.construct_simulation(integrator)

    def sample_x_from_equilibrium(self):
        """Draw sample (uniformly, with replacement) from cache of configuration samples"""
        return self.equilibrium_simulator.sample_x_from_equilibrium()

    def sample_v_given_x(self, x):
        """Sample velocities from (constrained) Maxwell-Boltzmann distribution."""
        self.simulation.context.setPositions(x)
        self.simulation.context.setVelocitiesToTemperature(self.equilibrium_simulator.temperature)
        self.simulation.context.applyVelocityConstraints(self.constraint_tolerance)
        return get_velocities(self.simulation)

    def accumulate_shadow_work(self, x_0, v_0, n_steps):
        """Run the integrator for n_steps and return the change in energy - the heat."""
        get_energy = lambda: get_total_energy(self.simulation)
        get_heat = lambda: self.simulation.integrator.getGlobalVariableByName("heat")

        set_positions(self.simulation, x_0)
        set_velocities(self.simulation, v_0)

        # Apply position and velocity constraints.
        self.simulation.context.applyConstraints(self.constraint_tolerance)
        self.simulation.context.applyVelocityConstraints(self.constraint_tolerance)

        E_0 = get_energy()
        Q_0 = get_heat()

        self.simulation.step(n_steps)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        return delta_E.value_in_unit(W_unit) - delta_Q

    def collect_protocol_samples(self, n_protocol_samples, protocol_length, marginal="configuration"):
        """Perform nonequilibrium measurements, aimed at measuring the free energy difference for the chosen marginal."""
        W_shads_F, W_shads_R = np.zeros(n_protocol_samples), np.zeros(n_protocol_samples)
        for i in tqdm(range(n_protocol_samples)):
            x_0 = self.sample_x_from_equilibrium()
            v_0 = self.sample_v_given_x(x_0)
            W_shads_F[i] = self.accumulate_shadow_work(x_0, v_0, protocol_length)

            x_1 = get_positions(self.simulation)
            if marginal == "configuration":
                v_1 = self.sample_v_given_x(x_1)
            elif marginal == "full":
                v_1 = get_velocities(self.simulation)
            else:
                raise NotImplementedError("`marginal` must be either 'configuration' or 'full'")

            if (np.isnan(x_1).sum() + np.isnan(v_1).sum()) > 0:
                W_shads_R *= np.nan
                W_shads_F *= np.nan
                print("NaNs encountered! Terminating early...")
                break

            W_shads_R[i] = self.accumulate_shadow_work(x_1, v_1, protocol_length)

            # if we've encountered any NaNs, terminate early
            if (np.isnan(W_shads_F).sum() + np.isnan(W_shads_R).sum()) > 0:
                W_shads_R *= np.nan
                W_shads_F *= np.nan
                print("NaNs encountered! Terminating early...")
                break

        return np.array(W_shads_F), np.array(W_shads_R)


if __name__ == "__main__":
    pass
