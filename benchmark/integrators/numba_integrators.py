import numpy as np
from numba import jit

def jit_if_possible(f):
    """If the function isn't already
    JIT-compiled, JIT it now."""
    try:
        f = jit(f)
    except:
        pass
    return f

def vvvr_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_vvvr(x0, v0, n_steps, gamma, dt):
        """Simulate n_steps of VVVR, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps), np.zeros(n_steps)
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

            # half R step
            x = x + ((dt/2.0) * v)

            # half R step
            x = x + ((dt/2.0) * v)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # O step
            ke_old = 0.5 * m * v ** 2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            xs[i] = x
            vs[i] = v

            # Update W_shads
            E_new = potential(x) + 0.5 * m * v ** 2
            W_shads[i] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_vvvr)

def orvro_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_orvro(x0, v0, n_steps, gamma, dt):
        """Simulate n_steps of orvro, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps), np.zeros(n_steps)
        xs[0] = x0
        vs[0] = v0
        E_old = potential(x) + 0.5 * m * v**2

        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # half O step
            ke_old = 0.5 * m * v**2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # half R step
            x = x + ((dt / 2.0) * v)

            # V step
            v = v + ((dt) * force(x) / m)

            # half R step
            x = x + ((dt / 2.0) * v)

            # half O step
            ke_old = 0.5 * m * v ** 2
            v = (a * v) + b * velocity_scale * np.random.randn()
            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            xs[i] = x
            vs[i] = v

            # Update W_shads
            E_new = potential(x) + 0.5 * m * v ** 2
            W_shads[i] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_orvro)

# TODO: simulator that just stores trajectory in terms of histogram to start with
# For now though, can just simulate in chunks.


def baoab_factory(potential, force, velocity_scale, m):
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_baoab(x0, v0, n_steps, gamma, dt):
        """Simulate n_steps of BAOAB, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps), np.zeros(n_steps)
        xs[0] = x0
        vs[0] = v0
        E_old = potential(x) + 0.5 * m * v**2

        # Mixing parameters for half O step
        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # R step
            x = x + ((dt / 2.0) * v)

            # half O step
            ke_old = 0.5 * m * v**2
            v = (a * v) + b * velocity_scale * np.random.randn()

            # half O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # R step
            x = x + ((dt / 2.0) * v)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            xs[i] = x
            vs[i] = v

            # Update W_shads
            E_new = potential(x) + 0.5 * m * v ** 2
            W_shads[i] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_baoab)

def aboba_factory(potential, force, velocity_scale, m):
    """R V O O V R"""
    potential = jit_if_possible(potential)
    force = jit_if_possible(force)

    def simulate_aboba(x0, v0, n_steps, gamma, dt):
        """Simulate n_steps of ABOBA, accumulating heat
        """
        Q = 0
        W_shads = np.zeros(n_steps)
        x, v = x0, v0
        xs, vs = np.zeros(n_steps), np.zeros(n_steps)
        xs[0] = x0
        vs[0] = v0
        E_old = potential(x) + 0.5 * m * v**2

        # Mixing parameters for half O step
        a = np.exp(-gamma * (dt / 2.0))
        b = np.sqrt(1 - np.exp(-2 * gamma * (dt / 2.0)))

        for i in range(1, n_steps):
            # R step
            x = x + ((dt / 2.0) * v)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # half O step
            ke_old = 0.5 * m * v**2
            v = (a * v) + b * velocity_scale * np.random.randn()

            # half O step
            v = (a * v) + b * velocity_scale * np.random.randn()

            ke_new = 0.5 * m * v ** 2
            Q += (ke_new - ke_old)

            # V step
            v = v + ((dt / 2.0) * force(x) / m)

            # R step
            x = x + ((dt / 2.0) * v)

            xs[i] = x
            vs[i] = v

            # Update W_shads
            E_new = potential(x) + 0.5 * m * v ** 2
            W_shads[i] = (E_new - E_old) - Q

        return xs, vs, Q, W_shads

    return jit(simulate_aboba)

def metropolis_hastings_factory(q):
    q = jit_if_possible(q)


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
