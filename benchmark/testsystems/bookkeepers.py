import gc
import os

import numpy as np
from benchmark.integrators.kyle.xchmc import XCGHMCIntegrator
from simtk import unit
import simtk.openmm as mm
from simtk.openmm import app
from tqdm import tqdm
from benchmark.integrators import LangevinSplittingIntegrator

from benchmark.utilities import strip_unit, get_total_energy, get_velocities, get_positions, \
    set_positions, set_velocities, remove_barostat, remove_center_of_mass_motion_remover, get_potential_energy

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

    def __init__(self, platform, topology, system, positions, temperature, xcghmc_timestep,
                 burn_in_length, n_samples, thinning_interval, name):

        self.platform = platform
        self.topology = topology
        self.system = system
        self.positions = positions
        self.temperature = temperature
        self.xcghmc_timestep = xcghmc_timestep
        self.burn_in_length = burn_in_length
        self.n_samples = n_samples
        self.thinning_interval = thinning_interval
        self.name = name
        self.cached = False
        self.constraint_tolerance = 1e-8

    def load_or_simulate_x_samples(self):
        """If we've already collected and stored equilibrium samples, load those
        Otherwise, collect equilibrium samples"""
        self._path_to_samples, self._path_to_box_vectors = self.get_path_to_samples_and_box_vectors()
        if self.check_for_cached_samples():
            print("Cache found: loading...")
            self.x_samples = self.load_equilibrium_samples()
        else:
            print("Cache not found: collecting equilibrium samples...")
            self.x_samples = self.collect_equilibrium_samples()
            self.save_equilibrium_samples(*self.x_samples)
        self.cached = True

    def get_acceptance_rate(self):
        """Return the number of acceptances divided by the number of proposals."""
        xchmc = self.unbiased_simulation.integrator
        return float(sum(xchmc.all_counts[:-1])) / sum(xchmc.all_counts)

    def construct_unbiased_simulation(self, use_reference=False):
        n_steps = 10
        return self.construct_simulation(
            XCGHMCIntegrator(temperature=self.temperature, steps_per_hmc=n_steps, extra_chances=15,
                             steps_per_extra_hmc=n_steps, timestep=self.xcghmc_timestep), use_reference=use_reference)

    def collect_equilibrium_samples(self):
        """Collect equilibrium samples, return as (n_samples, n_atoms, 3) numpy array"""

        print("Collecting equilibrium samples for '%s'..." % self.name)

        self.unbiased_simulation = self.construct_unbiased_simulation()
        set_positions(self.unbiased_simulation, self.positions)
        print("Minimizing...")
        self.unbiased_simulation.minimizeEnergy()

        print('"Burning in" unbiased sampler for {:.3}ps...'.format(
            (self.burn_in_length * self.xcghmc_timestep * 10).value_in_unit(unit.picoseconds)))
        for _ in tqdm(range(self.burn_in_length)):
            self.unbiased_simulation.step(1)
        print("Burn-in XC-GHMC acceptance rate: {:.3f}%".format(100 * self.get_acceptance_rate()))

        # Collect equilibrium samples
        print("Collecting equilibrium samples...")
        equilibrium_samples = []
        box_vectors = []
        for _ in tqdm(range(self.n_samples)):
            self.unbiased_simulation.step(self.thinning_interval)
            state = self.unbiased_simulation.context.getState(getPositions=True)
            x = state.getPositions(asNumpy=True)
            equilibrium_samples.append(strip_unit(x))
            box_vectors.append(state.getPeriodicBoxVectors())
        print("Equilibrated XC-GHMC acceptance rate: {:.3f}%".format(100 * self.get_acceptance_rate()))

        return np.array(equilibrium_samples), np.array(box_vectors)

    def get_path_to_samples_and_box_vectors(self):
        """Samples are {name}_samples.npy in DATA_PATH"""
        samples = os.path.join(DATA_PATH, '{}_samples.npy'.format(self.name))
        box_vectors = os.path.join(DATA_PATH, '{}_box_vectors.npy'.format(self.name))
        return samples, box_vectors

    def check_for_cached_samples(self):
        """Check if there's a file where we expect to find cached
        equilibrium samples.
        """
        # TODO : Need to check if any of the simulation parameters have changed.
        return os.path.exists(self._path_to_samples) and os.path.exists(self._path_to_box_vectors)

    def save_equilibrium_samples(self, x_samples, box_vectors):
        """Save numpy archive of equilibrium samples to disk."""
        np.save(self._path_to_samples, x_samples)
        np.save(self._path_to_box_vectors, box_vectors)

    def load_equilibrium_samples(self):
        """Load numpy archive of equilibrium samples"""
        print("Loading equilibrium samples from {}...".format(self._path_to_samples))
        x_samples = np.load(self._path_to_samples)
        print("Loading box vectors from {}...".format(self._path_to_box_vectors))
        x_box_vectors = np.load(self._path_to_box_vectors)
        return x_samples, x_box_vectors

    def sample_x_from_equilibrium(self):
        """Draw sample (uniformly, with replacement) from cache of configuration samples"""
        if self.cached == False:
            self.load_or_simulate_x_samples()

        i = np.random.randint(len(self.x_samples))
        return self.x_samples[0][i], self.x_samples[1][i]

    def sample_v_given_x(self, x):
        """Sample velocities from (constrained) Maxwell-Boltzmann distribution."""
        if not hasattr(self, "unbiased_simulation"):
            self.unbiased_simulation = self.construct_unbiased_simulation(use_reference=True)
        self.unbiased_simulation.context.setPositions(x)
        self.unbiased_simulation.context.setVelocitiesToTemperature(self.temperature)
        self.unbiased_simulation.context.applyVelocityConstraints(self.constraint_tolerance)
        return get_velocities(self.unbiased_simulation)

    def construct_simulation(self, integrator, use_reference=False):
        """Construct a simulation instance given an integrator."""
        gc.collect()  # make sure that any recently deleted Contexts actually get deleted...
        if use_reference:
            platform = mm.Platform.getPlatformByName('Reference')
        else:
            platform = self.platform
        simulation = app.Simulation(self.topology, self.system, integrator, platform)
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

    def accumulate_shadow_work(self, x_0, v_0, n_steps, box_vectors=None, store_potential_energy=False, store_W_shad_trace=False):
        """Run the integrator for n_steps and return the change in energy - the heat."""
        get_energy = lambda: get_total_energy(self.simulation).value_in_unit(W_unit)
        get_potential = lambda: get_potential_energy(self.simulation).value_in_unit(W_unit)
        get_heat = lambda: self.simulation.integrator.getGlobalVariableByName("heat")

        result = {}

        try:
            set_positions(self.simulation, x_0, box_vectors=box_vectors)
            set_velocities(self.simulation, v_0)
        except:
            print("Error setting positions or velocities!")
            return result

        # Apply position and velocity constraints.
        self.simulation.context.applyConstraints(self.constraint_tolerance)
        self.simulation.context.applyVelocityConstraints(self.constraint_tolerance)

        E_0 = get_energy()
        Q_0 = get_heat()

        if store_potential_energy or store_W_shad_trace:
            if store_potential_energy:
                potential_energies = [get_potential()]
            if store_W_shad_trace:
                total_energies = [E_0]
                heats = [Q_0]
                W_shad_trace = []
            for _ in range(n_steps):
                self.simulation.step(1)
                if store_potential_energy:
                    potential_energies.append(get_potential())
                if store_W_shad_trace:
                    total_energies.append(get_energy())
                    heats.append(get_heat())
                    DeltaE = total_energies[-1] - total_energies[-2]
                    DeltaQ = heats[-1] - heats[-2]
                    W_shad_trace.append(DeltaE - DeltaQ)
            if store_potential_energy:
                result["potential_energies"] = np.array(potential_energies)
            if store_W_shad_trace:
                result["W_shad_trace"] = np.array(W_shad_trace)
        else:
            self.simulation.step(n_steps)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        result["W_shad"] = delta_E - delta_Q

        return result

    def collect_protocol_samples(self, n_protocol_samples, protocol_length, marginal="configuration",
                                 store_potential_energy_traces=False, store_W_shad_traces=False):
        """Perform nonequilibrium measurements, aimed at measuring the free energy difference for the chosen marginal."""
        W_shads_F, W_shads_R = np.zeros(n_protocol_samples), np.zeros(n_protocol_samples)

        if marginal not in ["configuration", "full"]:
            raise NotImplementedError("`marginal` must be either 'configuration' or 'full'")

        potential_energy_traces = []
        W_shad_traces = []

        for i in tqdm(range(n_protocol_samples)):

            #try:
            x_0, box_vectors = self.sample_x_from_equilibrium()
            v_0 = self.sample_v_given_x(x_0)
            result_F = self.accumulate_shadow_work(x_0, v_0, protocol_length, box_vectors, store_W_shad_trace=store_W_shad_traces)
            W_shads_F[i] = result_F["W_shad"]

            x_1 = get_positions(self.simulation)
            if marginal == "configuration":
                v_1 = self.sample_v_given_x(x_1)
            elif marginal == "full":
                v_1 = get_velocities(self.simulation)

            result_R = self.accumulate_shadow_work(x_1, v_1, protocol_length,
                                                   store_potential_energy=store_potential_energy_traces, store_W_shad_trace=store_W_shad_traces)
            if len(result_R) == 0: # this means that self.accumulate_shadow_work tried to set coordinates to NaN
                W_shads_R *= np.nan
                W_shads_F *= np.nan
                print("Simulation crashed! Terminating early...")
                break

            W_shads_R[i] = result_R["W_shad"]
            if store_potential_energy_traces:
                potential_energy_traces.append(result_R["potential_energies"])
            if store_W_shad_traces:
                W_shad_traces.append((result_F["W_shad_trace"], result_R["W_shad_trace"]))


        result = {}
        result["W_shads_F"] = W_shads_F
        result["W_shads_R"] = W_shads_R
        if store_W_shad_traces:
            result["W_shad_traces"] = W_shad_traces
        if store_potential_energy_traces:
            result["potential_energies"] = potential_energy_traces

        return result


if __name__ == "__main__":
    pass
