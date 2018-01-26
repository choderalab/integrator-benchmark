# confirm that <exp[-w_shad]> = 1 for N steps of Langevin dynamics with any integrator, starting from pi

from quartic_kl_validation import NumbaNonequilibriumSimulator, schemes, equilibrium_simulator
from tqdm import tqdm
from functools import partial
import numpy as np

from quartic_kl_validation import gamma, n_protocol_samples, protocol_length, dt_range

n_samples = int(n_protocol_samples / 100)
results = {}

for scheme in schemes.keys():
    for timestep in dt_range:
        sim = NumbaNonequilibriumSimulator(equilibrium_simulator, partial(schemes[scheme], gamma=gamma, dt=timestep))

        W_shads = np.zeros(n_samples)
        for i in tqdm(range(n_samples)):
            x = sim.sample_x_from_equilibrium()
            v = sim.sample_v_given_x(x)

            W_shads[i] = sim.accumulate_shadow_work(x, v, protocol_length)

        results[(scheme, timestep)] = W_shads

        print(scheme, timestep)
        print("\t<exp[-W_shad]> = ",np.mean(np.exp(-W_shads)))

from pickle import dump
with open("1_wshad_sanity_check.pkl", "wb") as f:
    dump(results, f)