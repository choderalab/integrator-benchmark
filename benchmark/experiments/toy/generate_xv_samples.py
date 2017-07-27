import numpy as np
from benchmark import DATA_PATH
from benchmark.testsystems import quartic

potential = quartic.potential
force = quartic.force
velocity_scale = quartic.velocity_scale
m = quartic.mass

import os

from benchmark.integrators import baoab_factory, vvvr_factory
gamma = 10

n_steps = 10000000
n_thinning = 5
x_0, v_0 = 0, np.random.randn()
timesteps_to_try = np.array([0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

baoab = baoab_factory(potential=potential, force=force, velocity_scale=velocity_scale, m=m)
vvvr = vvvr_factory(potential=potential, force=force, velocity_scale=velocity_scale, m=m)


def sample_xv(scheme="VVVR"):
    if scheme == "VVVR":
        integrator = vvvr
    elif scheme == "BAOAB":
        integrator = baoab
    else:
        print("Nooope.")
        exit()

    for i, dt in enumerate(timesteps_to_try):
        print(dt)
        print("\nTesting {} with timestep dt={}".format(scheme, dt))

        xs, vs = [], []
        # these have the following signature: (x0, v0, n_steps, gamma, dt)
        for _ in range(n_thinning):
            xs_, vs_, Q, W_shad = integrator(x_0, v_0, n_steps, gamma, dt)
            xs_ = xs_[100:]
            vs_ = vs_[100:]

            if len(np.where(np.isnan(xs_))[0]) > 0:
                max_ind = np.where(np.isnan(xs_))[0][0]
                xs_ = xs_[:max_ind - 10]
                vs_ = vs_[:max_ind - 10]

            xs.append(xs_[::n_thinning])
            vs.append(vs_[::n_thinning])

        xs = np.hstack(xs)
        vs = np.hstack(vs)

        xv = np.vstack((xs, vs)).T
        np.save(os.path.join(DATA_PATH, "quartic_xv_{}_{}.npy".format(scheme, dt)), xv)


if __name__ == "__main__":
    np.random.seed(1)

    sample_xv("VVVR")
    sample_xv("BAOAB")

