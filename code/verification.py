from sympy import *
from functools import partial


class State():
    def __init__(self, x, v):
        self.x, self.v = x,v
    def __repr__(self):
        return "({}, {})".format(self.x, self.v)

m = symbols('m')
gamma = symbols('gamma')
f = symbols('f')
kT = symbols('kT')
U = symbols('U')

def total_energy(state):
    """Computes the total energy of a state"""
    return U(state.x) + (0.5 * m * state.v**2)

def count_steps(splitting="OVRVO"):
    """Computes the number of O, R, V steps in the splitting string"""
    n_O = sum([step == "O" for step in splitting])
    n_R = sum([step == "R" for step in splitting])
    n_V = sum([step == "V" for step in splitting])
    return n_O, n_R, n_V

def construct_traj(splitting="OVRVO"):
    """Computes a list of states of length len(splitting)+1.

    For deterministic steps R and V, we compute the next state as a deterministic function
    of the current state.

    For stochastic steps "O", we give a new variable name to the next state.

    For example, the trajectory produced by "RVO" would be
    [
    (x_0, v_0),
    (x_0 + v_0 * h, v_0),
    (x_0 + v_0 * h, v_0 + h * f(x_0 + v_0 * h) / m)
    (x_0 + v_0 * h, v_4)
    ]
    """
    f, h, m = symbols("f h m")
    n_O, n_R, n_V = count_steps(splitting)

    bath_variable_names = "xi"

    traj = [State(*symbols("x_0 v_0"))]

    a = exp(-gamma * h / n_O)
    b = sqrt(1 - a ** 2)

    for i, step in enumerate(splitting):
        next_x, next_v = traj[-1].x, traj[-1].v

        if step == "O":
            next_v = a * traj[-1].v + b * symbols("{}_{}".format(
                bath_variable_names, i))
        elif step == "R":
            next_x = traj[-1].x + (traj[-1].v) * (h / n_R)
        elif step == "V":
            next_v = traj[-1].v + (f(traj[-1].x) / m) * (h / n_V)

        traj.append(State(next_x, next_v))

    return traj

def construct_reverse_traj(splitting="OVRVO"):
    """Computes a list of states of length len(splitting)+1.

    For deterministic steps R and V, we compute the next state as a deterministic function
    of the current state.

    For stochastic steps "O", we give a new variable name to the next state.

    For example, the trajectory produced by "RVO" would be
    [
    (x_0, v_0),
    (x_0 + v_0 * h, v_0),
    (x_0 + v_0 * h, v_0 + h * f(x_0 + v_0 * h) / m)
    (x_0 + v_0 * h, v_4)
    ]
    """
    f, h, m = symbols("f h m")
    n_O, n_R, n_V = count_steps(splitting)

    bath_variable_names = r"\tilde{xi}"
    n_steps = len(splitting)

    traj = [State(symbols("x_{}".format(n_steps)), - symbols("v_{}".format(n_steps)))]

    a = exp(-gamma * h / n_O)
    b = sqrt(1 - a ** 2)

    for i in range(len(splitting))[::-1]:
        step = splitting[i]
        next_x, next_v = traj[-1].x, traj[-1].v

        if step == "O":
            next_v = - a * traj[-1].v + b * symbols("{}_{}".format(
                bath_variable_names, i))
            #next_v = symbols("v_{}".format(i + 1))
        elif step == "R":
            next_x = traj[-1].x - (traj[-1].v) * (h / n_R)
        elif step == "V":
            next_v = - traj[-1].v + (f(traj[-1].x) / m) * (h / n_V)

        traj.append(State(next_x, next_v))

    return traj

def compute_substep_energy_changes(traj):
    """Compute the energy change during each substep."""
    Es = [total_energy(state) for state in traj]
    return [Es[i+1] - Es[i] for i in range(len(Es)-1)]

def normal_pdf(x, mu, sigma):
    """Gaussian probability density function of x given parameters mu and sigma."""
    return (1 / (sqrt(2 * pi * sigma ** 2))) * exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def R_prob(from_state, to_state, dt):
    """Probability that to_state follows from_state, after
    a deterministic position update."""
    if (from_state.v == to_state.v) and\
            (to_state.x == (from_state.x + from_state.v * dt)):
        return 1
    else:
        return 0


def V_prob(from_state, to_state, dt):
    """Probability that to_state follows from_state, after
        a deterministic velocity update."""

    if (from_state.x == to_state.x) and\
            (to_state.v == (from_state.v + f(from_state.x) * dt / m)):
        return 1
    else:
        return 0


def O_prob(from_state, to_state, dt):
    """Probability that to_state follows from_state, after a stochastic
     velocity update.
     A Gaussian centered on a * from_state.v, with sigma=sqrt(1-a^2) sqrt(kT / m).
     In limit gamma -> + \infty, a -> 0 and sigma -> sqrt(kT/m)
     In limit gamma -> 0, a -> 1 and sigma -> 0.
     """
    a = exp(-gamma * dt)

    sigma = sqrt(1 - a ** 2) * sqrt(kT / m)
    mu = a * from_state.v

    if (from_state.x == to_state.x):
        return normal_pdf(to_state.v, mu, sigma)
    else:
        return 0


def V_rand_prob(from_state, to_state, dt=None, fixed_v_density=symbols("q")):
    """Velocity randomization kernel:
        keep x the same
        ignore current v
        ignore dt
        sample next v i.i.d. from fixed distribution with density q

    Note:
        x-marginal in to_state ensemble = x-marginal in from_state ensemble
        v-marginal in to_state ensemble = q
    """

    if (from_state.x == to_state.x):
        return fixed_v_density(to_state.v)
    else:
        return 0

step_mapping = {"O": O_prob, "R": R_prob, "V": V_prob}

def reverse_kernel(transition_kernel):
    """Switch the order of "from_state" and "to_state" """

    def reversed_kernel(from_state, to_state, dt):
        return transition_kernel(to_state, from_state, dt)

    return reversed_kernel

def make_step_length_dict(splitting):
    h = symbols("h")
    n_O, n_R, n_V = count_steps(splitting)
    step_length = dict()
    step_length["O"], step_length["R"], step_length["V"] = h / n_O, h / n_R, h / n_V
    return step_length

def construct_forward_protocol(splitting="OVRVO"):
    """Return a list of transition kernels.

    step_mapping is a dict mapping from the letters "O", "R", "V" to parameterized transition densities,
        accepting from_state, to_state, and dt

    We fix the dt for each transition density appropriately, using the step_length dictionary
    """
    step_length = make_step_length_dict(splitting)
    protocol = []
    for step in splitting:
        transition_density = partial(step_mapping[step], dt=step_length[step])
        protocol.append(transition_density)

    return protocol


def construct_reverse_protocol(splitting="OVRVO"):
    """Run the steps in the reverse order, and for each step, use the time-reverse of that kernel."""
    step_length = make_step_length_dict(splitting)
    protocol = []
    for step in splitting[::-1]:
        transition_density = partial(reverse_kernel(step_mapping[step]), dt=step_length[step])
        protocol.append(transition_density)

    return protocol


def evaluate_path_probability(trajectory, protocol, verbose=True):
    """Evaluate the probability of a trajectory, given a protocol.
    Trajectory is one longer than protocol"""

    step_probs = []
    for i in range(len(protocol)):
        step_probs.append(protocol[i](trajectory[i], trajectory[i + 1]))
        if verbose:
            print("Probability of transitioning from {} to {}:".format(trajectory[i], trajectory[i+1]))
            print("\t{}\n".format(step_probs[-1]))

    return prod(step_probs)

def log_prob_xi(xi):
    """Evaluate the log probability of a given bath variable."""
    sigma = sqrt(kT / m)
    return - (xi**2 / sigma**2) - log(1 / (sqrt(2 * pi * sigma ** 2)))

def log_prob_bath_variables(trajectory, bath_variable_name="xi"):
    """Take the sum of the log probabilities of each bath variable"""


    states_as_equations = []
    bath_variables = []

    for i in range(trajectory):
        state = trajectory[i]
        if symbol(bath_variable_name) in state.x:
            state_name = symbol("x_{}".format(i))
            definition = state.x

            #solve(Eq(state_name, definition), bath_variable_name)

        if  bath_variable_name in state.v:
            state_name = symbol("v_{}".format(i))
            definition = state.v

            states_as_equations.append(Eq(state_name, definition))

    bath_variables = [solve_linear_system(states_as_equations, bath_variables)]

    return sum([log_prob_xi(bath_variable) for bath_variable in bath_variables])


def flip_velocity_signs(trajectory):
    return [State(s.x, -s.v) for s in trajectory]


def compute_DeltaE_decomposition(splitting):
    """DeltaE = W_shad + heat
    DeltaE = final energy - initial energy
    heat = sum([(energy after O step - energy before O step) for O step in splitting])
    W_shad = DeltaE - heat
    """

    energies = [total_energy(State(*symbols("x_{} v_{}".format(i, i)))) for i in range(len(splitting) + 1)]
    heat = sum([energies[i+1] - energies[i] for i in range(len(splitting)) if splitting[i] == "O"])
    DeltaE = energies[-1] - energies[0]
    W_shad = DeltaE - heat

    return DeltaE, W_shad, heat

def CFT_work(log_forward_prob, log_reverse_prob):
    return log_forward_prob - log_reverse_prob

if __name__ == "__main__":
    splitting = "OVRVO"
    n_steps = len(splitting)
    init_printing()

    DeltaE, W_shad, heat = compute_DeltaE_decomposition(splitting)

    forward_trajectory = construct_traj(splitting)
    reverse_trajectory = construct_reverse_traj(splitting)
    #reverse_trajectory = flip_velocity_signs(construct_reverse_traj(splitting))
    #reverse_trajectory = construct_reverse_trajectory(forward_trajectory)

    forward_protocol = construct_forward_protocol(splitting)
    reverse_protocol = construct_reverse_protocol(splitting)

    print("Evaluating forward probability")
    forward_prob = evaluate_path_probability(forward_trajectory, forward_protocol)
    print("\nEvaluating reverse probability")
    reverse_prob = evaluate_path_probability(reverse_trajectory, reverse_protocol)

    print("\nForward_prob / reverse_prob:")
    print((forward_prob / reverse_prob).simplify())
