"""
How big of a difference is there, on average, between taking a large 
geodesic drift step, and taking several smaller geodesic drift steps?


In other words, what does
$$f(n, dt) \equiv \mathbb{E}_\pi \| R(dt)(x, v) - (R^n(dt / n)(x,v)) \|$$
look like?
"""

import os
import pickle

import matplotlib
import numpy as np
from benchmark import DATA_PATH
from benchmark.testsystems import alanine_constrained, waterbox_constrained, src_constrained
from simtk import unit
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.utilities import stderr
from benchmark import FIGURE_PATH
from benchmark import simulation_parameters
from benchmark.utilities.openmm_utilities import get_masses

class GeodesicDrifter():
    def __init__(self, simulator, tolerance=simulation_parameters["tolerance"]):
        self.system = simulator.system
        self.simulation = simulator.unbiased_simulation
        self.tolerance = tolerance
        m = get_masses(self.system)
        self.m = np.ones((len(m), 3))
        for i in range(len(m)):
            self.m[i] *= m[i]
        self.m = self.m * m.unit

    def set_x(self, x):
        self.simulation.context.setPositions(x)

    def set_v(self, v):
        self.simulation.context.setVelocities(v)

    def get_x(self):
        return self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

    def get_v(self):
        return self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def get_f(self):
        return self.simulation.context.getState(getForces=True).getForces(asNumpy=True)

    def get_xv(self):
        return self.get_x(), self.get_v()

    def constrainPositions(self):
        self.simulation.context.applyConstraints(self.tolerance)

    def constrainVelocities(self):
        self.simulation.context.applyVelocityConstraints(self.tolerance)

    def R_step(self, x, v, h):
        x = x + h * v # (pre-constraint)
        self.set_x(x)
        self.constrainPositions()
        x1 = self.get_x() # (post-constraint)

        self.set_v(v + ((x - x1) / h))
        self.constrainVelocities()
        return self.get_xv()

    def V_step(self, f, v, h):
        self.set_v(v + h * f / self.m)
        self.constrainVelocities()
        return self.get_v()

    def drift(self, x, v, dt=1.0 * unit.femtosecond, n_steps=1):
        for _ in range(n_steps):
            x, v = self.R_step(x, v, dt / n_steps)

        return x.value_in_unit(x.unit)#, v.value_in_unit(v.unit) # TODO: Analysis that looks at v, too?

    def kick(self, x, v, dt=1.0 * unit.femtosecond, n_steps=1):
        self.set_x(x)
        f = self.get_f()

        for _ in range(n_steps):
            v = self.V_step(f, v, dt / n_steps)

        return v.value_in_unit(v.unit)

if __name__ == "__main__":
    n_samples = 10

    system_name = "waterbox_constrained"
    equilibrium_simulator = waterbox_constrained
    x_unit = equilibrium_simulator.unbiased_simulation.context.getState(getPositions=True).getPositions().unit

    # define some convenience functions...
    def sample_xv():
        """Sample positions and velocities from equilibrium"""
        x = equilibrium_simulator.sample_x_from_equilibrium() * x_unit
        # TODO: remove when bookkeeper class is refactored
        # ^ * x_unit in that line is only necessary because I stored the samples without units...
        v = equilibrium_simulator.sample_v_given_x(x)
        return x, v

    def distance(x, y):
        """Compute relevant distance between configurations x and y.
        For now, just the norm -- could do RMSD or something here instead..."""
        return np.linalg.norm(x - y)

    target_filename = "geodesic_drift_difference_{}".format(system_name)
    n_geodesic_steps = list(range(1, 10)) + list(range(10, 51)[::10])

    ys = {}
    yerrs = {}
    distances = {}
    timesteps = np.linspace(0.1, 10.0, 20) * unit.femtosecond

    geodesic_drifter = GeodesicDrifter(equilibrium_simulator)

    xs_drift = np.zeros((n_samples, len(timesteps), len(n_geodesic_steps)), dtype=object)
    xs_kick = np.zeros((n_samples, len(timesteps), len(n_geodesic_steps)), dtype=object)

    for i in tqdm(range(n_samples)):
        x, v = sample_xv()
        for j in range(len(timesteps)):
            for k in range(len(n_geodesic_steps)):
                xs_drift[i,j,k] = geodesic_drifter.drift(x, v, dt=timesteps[j], n_steps=n_geodesic_steps[k])
                xs_kick[i, j, k] = geodesic_drifter.kick(x, v, dt=timesteps[j], n_steps=n_geodesic_steps[k])


    discrepancy_curves_y = {}
    discrepancy_curves_yerr = {}
    baseline_k = len(n_geodesic_steps) - 1 # or 0
    # if baseline_k = 0, then we're comparing with a single large R step
    # if baseline_k = len(n_geodesic_steps) -1, then we're comparing to the largest number of
    #  drift steps tested, i.e. the closest approximation to exact geodesic drift
    test_k = sorted(list(set(range(len(n_geodesic_steps))) - {baseline_k}))

    def generate_discrepancy_curves(xs):
        for k in test_k:
            discrepancy_curves_y[k] = []
            discrepancy_curves_yerr[k] = []
            for j in range(len(timesteps)):
                distances = []
                for i in range(n_samples):
                    distances.append(distance(xs[i,j,k], xs[i,j,baseline_k]))
                discrepancy_curves_y[k].append(np.mean(distances))
                discrepancy_curves_yerr[k].append(stderr(distances))
        return discrepancy_curves_y, discrepancy_curves_yerr

    # pickle results
    results = {"x: timesteps": timesteps,
               "xs_drift[sample_ind, timestep_ind, drift_step_ind]": xs_drift,
               "xs_kcik[sample_ind, timestep_ind, drift_step_ind]": xs_kick}
    with open(os.path.join(DATA_PATH, target_filename + ".pkl"), "wb") as f:
        pickle.dump(results, f)

    # now, generate plot: one curve per n_geodesic_steps, x axis timestep, y axis discrepancy
    def plot(mode="drift", yscale="log"):
        if mode == "drift":
            xs = xs_drift
            coords = "x"
        elif mode == "kick":
            xs = xs_kick
            coords = "v"
        else:
            raise(Exception("Mode must be either 'drift' or 'kick'"))

        discrepancy_curves_y, discrepancy_curves_yerr = generate_discrepancy_curves(xs)

        x = timesteps.value_in_unit(unit.femtosecond)
        plt.figure()
        for k in test_k:
            plt.errorbar(x, discrepancy_curves_y[k], yerr=discrepancy_curves_yerr[k], label=n_geodesic_steps[k])


        plt.title(r"Average discrepancy in {}".format(coords))
        plt.xlabel("Timestep (fs)")
        plt.yscale(yscale)
        plt.legend(title=r"Number of geodesic {} steps $n$".format(mode))
        plt.savefig(os.path.join(FIGURE_PATH, target_filename + "_{}_{}.png".format(mode, yscale)))
        plt.close()


    # plot with linear y scale
    for mode in ["drift", "kick"]:
        for yscale in ["log", "linear"]:
            plot(mode, yscale)


# TODO: how does error depend on the location within the integrator substep?
