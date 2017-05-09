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
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.testsystems import alanine_constrained, waterbox_constrained
from simtk import unit
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.utilities import stderr
from benchmark import FIGURE_PATH

if __name__ == "__main__":
    n_samples = 1000

    system_name = "alanine_constrained"
    equilibrium_simulator = alanine_constrained


    # define some convenience functions...
    def sample_xv():
        """Sample positions and velocities from equilibrium"""
        x = equilibrium_simulator.sample_x_from_equilibrium()
        v = equilibrium_simulator.sample_v_given_x(x)
        return x, v


    def set_xv(sim, x, v):
        """Set positions and velocities"""
        sim.context.setPositions(x)
        sim.context.setVelocities(v)


    def get_x(sim):
        """Get current (unitless)"""
        x = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        return x.value_in_unit(x.unit)


    def distance(x, y):
        """Compute relevant distance between configurations x and y.
        For now, just the norm -- could do RMSD or something here instead..."""
        return np.linalg.norm(x - y)


    target_filename = "geodesic_drift_difference_{}".format(system_name)

    schemes = {}
    for i in range(2, 10):
        schemes[i] = " ".join(["R"] * i)

    ys = {}
    yerrs = {}
    distances = {}
    timesteps = np.linspace(1.0, 10.0, 20) * unit.femtosecond

    # TODO: Optimize performance: don't actually have to use a CustomIntegrator at all here
    # (This is probably slower than it needs to be, because forces are evaluated at each step.
    # Could replace with a function that just does the geodesic drift steps without invoking integrator
    # applying constraints via OpenMM's Python API sim.context.applyVelocityConstraints(tol)...)

    for name, scheme in schemes.items():
        print("Number of geodesic drift steps n={}".format(name))
        ys[name] = []
        yerrs[name] = []
        distances[name] = []

        for dt in tqdm(timesteps):
            baseline = equilibrium_simulator.construct_simulation(
                LangevinSplittingIntegrator(splitting="R", timestep=dt, override_splitting_checks=True))

            lsi = LangevinSplittingIntegrator(splitting=scheme, timestep=dt,
                                              override_splitting_checks=True)
            test = equilibrium_simulator.construct_simulation(lsi)
            distances[name].append([])
            current_distances = distances[name][-1]

            for _ in range(n_samples):
                # draw (x,v) from equilibrium
                x, v = sample_xv()
                set_xv(baseline, x, v)
                set_xv(test, x, v)

                # take a single large R step (baseline) or several small R steps (test)
                baseline.step(1)
                test.step(1)

                x_baseline = get_x(baseline)
                x_test = get_x(test)
                current_distances.append(distance(x_baseline, x_test))


            ys[name].append(np.mean(current_distances))
            yerrs[name].append(stderr(current_distances))

            # TODO: Just store full snapshots? (e.g. `xs[name].append((x_baseline, x_test))`)

    # pickle results
    results = {"x: timesteps": timesteps,
               "y:  discrepancy": ys,
               "yerr: standard error": yerrs,
               "raw: distances": distances}
    with open(os.path.join(DATA_PATH, target_filename + ".pkl", "wb")) as f:
        pickle.dump(results, f)

    # now, generate plot: one curve per n_drift_steps, x axis timestep, y axis discrepancy
    plt.figure()
    x = timesteps.value_in_unit(unit.femtosecond)
    for s in schemes:
        plt.errorbar(x, ys[s], yerr=yerrs[s], label=s)
    plt.title(r"Average discrepancy between $R(x,v)$ and $R^n(x,v)$")
    plt.xlabel("Timestep (fs)")
    plt.ylabel(r"$\mathbf{E}_{\pi} \| R_{dt}(x,v) - R_{dt}^n(x,v) \|$")
    plt.legend(title=r"Number of geodesic drift steps $n$")
    plt.savefig(os.path.join(FIGURE_PATH, target_filename + ".png"))
    plt.close()

# TODO: how does error depend on the location within the integrator substep?
# TODO: Instead of using R_{dt} as the baseline, what if we use R^10_{dt/10} as the baseline?
# (I.e.: use a close approximation to exact geodesic drift as baseline...)
