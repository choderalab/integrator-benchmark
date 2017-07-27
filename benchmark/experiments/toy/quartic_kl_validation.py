# In this experiment, we validate the nonequilibrium estimator of configuration-marginal KL divergence
# on three schemes: ABOBA, BAOAB, and OBABO


# 1. Collect samples from steady state for each scheme (exact, BAOAB, ABOBA, OBABO)
# 2. Estimate KL divergences using "ground-truth" numerical methods (using histograms)
# 3. Estimate KL divergences using nonequilibrium scheme

# collect steady state samples


import benchmark.testsystems
from benchmark.testsystems import NumbaNonequilibriumSimulator
from benchmark.integrators.langevin import LangevinSplittingIntegrator
from benchmark import DATA_PATH
import os
import pickle
import numpy as np
from simtk import unit
from benchmark.plotting import plot_scheme_comparison
from benchmark.evaluation.analysis import estimate_nonequilibrium_free_energy
from benchmark.integrators import baoab_factory, vvvr_factory, aboba_factory
from functools import partial


if __name__ == "__main__":
    n_protocol_samples, protocol_length = 500000, 25
    system_name = "quartic"
    equilibrium_simulator = benchmark.testsystems.quartic
    target_filename = os.path.join(DATA_PATH, "scheme_comparison_{}.pkl".format(system_name))

    potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
    schemes = {"BAOAB": baoab_factory(potential, force, velocity_scale, mass),
               "VVVR": vvvr_factory(potential, force, velocity_scale, mass),
               #"ABOBA": aboba_factory(potential, force, velocity_scale, mass),
               }
    timesteps = np.array([0.5,0.9]) #np.linspace(0.5, 0.9, 2)
    gamma = 0.5
    noneq_simulators = {}
    for name, scheme in schemes.items():
        for timestep in timesteps:
            # output of the integrator factory is a function with this signature: simulate_aboba(x0, v0, n_steps, gamma, dt)
            #integrator = lambda x_0, v_0, n_steps : scheme(x_0, v_0, n_steps, gamma, timestep)
            noneq_simulators[(name, timestep)] = NumbaNonequilibriumSimulator(equilibrium_simulator,
                                                                              partial(scheme, gamma=gamma, dt=timestep))

    _ = equilibrium_simulator.sample_x_from_equilibrium()
    edges = np.histogram(equilibrium_simulator.x_samples, bins=50)[1]


    # we'll also want to plot the distance from equillibrium, which will require saving xs and vs, or at least their histograms
    # alrighty, let's update to save vs also...

    results = {}
    for marginal in ["configuration", "full"]:
        results[marginal] = {}
        for name, simulator in noneq_simulators.items():
            print(marginal, name)
            W_shads_F, W_shads_R, xs_F, xs_R = simulator.collect_protocol_samples(
                n_protocol_samples, protocol_length, marginal)

            # summarize xs_F and xs_R into histograms:
            xs_F_hists = np.array([np.histogram(np.nan_to_num(xs_F[:, i]), bins=edges)[0] for i in range(xs_F.shape[1])])
            xs_R_hists = np.array([np.histogram(np.nan_to_num(xs_R[:, i]), bins=edges)[0] for i in range(xs_R.shape[1])])

            results[marginal][name] = np.array(W_shads_F, dtype=np.float16),\
                                      np.array(W_shads_R, dtype=np.float16),\
                                      edges, xs_F_hists, xs_R_hists

            DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(W_shads_F[:,-1], W_shads_R[:,-1])
            print("\t{:.5f} +/- {:.5f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))

    with open(target_filename, "wb") as f:
        pickle.dump(results, f)

    plot_scheme_comparison(target_filename, system_name)
