import numpy as np
from numba import jit

def vvvr_factory(potential, force, velocity_scale, m):

    def simulate_vvvr(x0, v0, n_steps, gamma, dt, thinning_factor=1):
        """Simulate n_steps of VVVR, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps / thinning_factor)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps / thinning_factor), np.zeros(n_steps / thinning_factor)
        xs[0] = x0
        vs[0] = v0
        E_old = potential(x) + 0.5 * m * v**2

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # O step
            ke_old = 0.5 * m * v**2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # R step
            x = x + (dt * v)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # O step
            ke_old = 0.5 * m * v ** 2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # store
            if i % thinning_factor == 0:
                xs[i / thinning_factor] = x
                vs[i / thinning_factor] = v

                E_new = potential(x) + 0.5 * m * v ** 2
                W_shads[i / thinning_factor] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_vvvr)


def baoab_factory(potential, force, velocity_scale, m):
    def simulate_baoab(x0, v0, n_steps, gamma, dt, thinning_factor=1):
        """Simulate n_steps of BAOAB, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps / thinning_factor)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps / thinning_factor), np.zeros(n_steps / thinning_factor)
        xs[0] = x0
        vs[0] = v0
        E_old = potential(x) + 0.5 * m * v**2

        a = np.exp(-gamma * (dt))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt)))

        for i in range(1, n_steps):
            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # R step
            x = x + ((dt / 2.0) * v)

            # O step
            ke_old = 0.5 * m * v**2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # R step
            x = x + ((dt / 2.0) * v)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # store
            if i % thinning_factor == 0:
                xs[i / thinning_factor] = x
                vs[i / thinning_factor] = v

                E_new = potential(x) + 0.5 * m * v**2
                W_shads[i / thinning_factor] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_baoab)

def metropolis_hastings_factory(q):
    # If the unnormalized density function isn't already
    # JIT-compiled, JIT it now.
    try:
        q = jit(q)
    except:
        pass


    def rw_metropolis_hastings(x0, n_steps):
        xs = np.zeros(n_steps)
        xs[0] = x0

        # draw all the random numbers we'll need
        proposal_eps = np.random.randn(n_steps) # standard normal
        accept_eps = np.random.rand(n_steps) # uniform(0,1)

        for i in range(1, n_steps):
            x_prop = xs[i-1] + proposal_eps[i]
            a_r_ratio = q(x_prop) / q(xs[i-1])

            # accept / reject
            if a_r_ratio > accept_eps[i]:
                xs[i] = x_prop
            else:
                xs[i] = xs[i-1]
        return xs

    return jit(rw_metropolis_hastings)
