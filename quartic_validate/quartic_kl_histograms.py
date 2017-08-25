# Collect many samples from noneq steady state of each integrator over a range of dt.
# Summarize the samples into 1D and 2D histograms and save them as .npz files.

from quartic_kl_validation import NumbaBookkeepingSimulator, dt_range, gamma
from numba import jit
import numpy as np
system_name = "quartic"

equilibrium_simulator = NumbaBookkeepingSimulator()


timestep = max(dt_range)


def jit_if_possible(f):
    """If the function isn't already
    JIT-compiled, JIT it now."""
    try:
        f = jit(f)
    except:
        pass
    return f

# these integrators return (x,v) trajectories
def ovrvo_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_ovrvo(x0, v0, n_steps, dt):
        x, v = x0, v0
        xs = np.zeros(n_steps)
        vs = np.zeros(n_steps)
        xs[0] = x
        vs[0] = v

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + (dt * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            xs[i] = x
            vs[i] = v

        return xs, vs

    return jit(simulate_ovrvo)


def orvro_factory(potential, force, velocity_scale, m):
    force = jit_if_possible(force)

    def simulate_orvro(x0, v0, n_steps, dt):
        x, v = x0, v0
        xs = np.zeros(n_steps)
        vs = np.zeros(n_steps)
        xs[0] = x
        vs[0] = v

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            xs[i] = x
            vs[i] = v

        return xs, vs

    return jit(simulate_orvro)


def vrorv_factory(potential, force, velocity_scale, m):
    force = jit_if_possible(force)

    def simulate_vrorv(x0, v0, n_steps, dt):
        x, v = x0, v0
        xs = np.zeros(n_steps)
        vs = np.zeros(n_steps)
        xs[0] = x
        vs[0] = v

        a = np.exp(-gamma * (dt / 1.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 1.0)))

        for i in range(1, n_steps):
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)

            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            xs[i] = x
            vs[i] = v

        return xs, vs

    return jit(simulate_vrorv)


def rvovr_factory(potential, force, velocity_scale, m):
    force = jit_if_possible(force)

    def simulate_rvovr(x0, v0, n_steps, dt):
        x, v = x0, v0
        xs = np.zeros(n_steps)
        vs = np.zeros(n_steps)
        xs[0] = x
        vs[0] = v

        a = np.exp(-gamma * (dt / 1.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 1.0)))

        for i in range(1, n_steps):
            # R step
            x = x + ((dt / 2.0) * v)
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # O step
            v = (a * v) + b * velocity_scale * np.random.randn()
            # V step
            v = v + ((dt / 2.0) * force(x) / m)
            # R step
            x = x + ((dt / 2.0) * v)

            xs[i] = x
            vs[i] = v

        return xs, vs

    return jit(simulate_rvovr)



potential, force, velocity_scale, mass = equilibrium_simulator.potential, equilibrium_simulator.force, equilibrium_simulator.velocity_scale, equilibrium_simulator.mass
schemes = {"VRORV": vrorv_factory(potential, force, velocity_scale, mass),
           "RVOVR": rvovr_factory(potential, force, velocity_scale, mass),
           "ORVRO": orvro_factory(potential, force, velocity_scale, mass),
           "OVRVO": ovrvo_factory(potential, force, velocity_scale, mass),
           }


from tqdm import tqdm

data_range = (-2.5, 2.5)
n_bins = 50
bin_edges = np.linspace(data_range[0], data_range[1], num=n_bins)
one_d_hist_args = {"bins": n_bins * 2, "range": data_range}

def get_one_d_hist(xs):
    sampled_x_hist, _ = np.histogram(xs, **one_d_hist_args)
    return sampled_x_hist

def get_two_d_hist(xs, vs):
    joint_sampled_dist, _, _ = np.histogram2d(xs, vs, bins=bin_edges)
    return joint_sampled_dist

def xv_to_counts(xs, vs):
    one_d_hist = get_one_d_hist(xs)
    two_d_hist = get_two_d_hist(xs, vs)

    return one_d_hist, two_d_hist

def get_large_sample_histogram(dt=1.0, scheme="VRORV"):

    counts_1d, counts_2d = 0, 0
    for _ in tqdm(range(1000)):
        xs, vs = schemes[scheme](np.random.randn(), np.random.randn(), 1000000, dt)

        # exclude NaNs
        if (np.isnan(xs).sum() + np.isnan(vs).sum()) == 0:
            one_d_hist, two_d_hist = xv_to_counts(xs[1000:], vs[1000:])
            counts_1d += one_d_hist
            counts_2d += two_d_hist
    return counts_1d, counts_2d


experiments = []
for dt in dt_range:
    for splitting in schemes.keys():
        experiments.append((dt, splitting))
print("# experiments: {}".format(len(experiments)))

if __name__ == "__main__":
    import sys

    try:
        job_id = int(sys.argv[1]) - 1
        np.random.seed(job_id)
    except:
        print("no input received!")
        job_id = np.random.randint(len(experiments))


    dt, scheme = experiments[job_id]
    print("dt: {}, scheme: {}".format(dt, scheme))
    print("simulating...")
    counts_1d, counts_2d = get_large_sample_histogram(dt, scheme)
    description = np.array([dt, scheme])

    experiment_name = "0_quartic"
    filename = "quartic_results/{}_{}.npz".format(experiment_name, job_id)

    print("saving to {}".format(filename))
    np.savez(filename,
             description=description,
             counts_1d=counts_1d,
             counts_2d=counts_2d)
