import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from simtk.openmm import app
from simtk import unit
import simtk.openmm as mm
import numpy as np
from openmmtools.testsystems import WaterBox, AlanineDipeptideVacuum
W_unit = unit.kilojoule_per_mole

figure_directory = "../figures/"
figure_format = ".jpg"

def savefig(name):
    plt.savefig("{}{}{}".format(figure_directory, name, figure_format), dpi=300)

def generate_solvent_solute_splitting_string(base_integrator="VRORV", K_p=1, K_r=3):
    """Generate string representing sequence of V0, V1, R, O steps, where force group 1
    is assumed to contain fast-changing, cheap-to-evaluate forces, and force group 0
    is assumed to contain slow-changing, expensive-to-evaluate forces.



    Currently only supports solvent-solute splittings of the VRORV (BAOAB / g-BAOAB)
    integrator, but it should be easy also to support splittings of the ABOBA integrator.

    Parameters
    -----------
    base_integrator: string
        Currently only supports VRORV
    K_p: int
        Number of times to evaluate force group 1 per timestep.
    K_r: int
        Number of inner-loop iterations

    Returns
    -------
    splitting_string: string
        Sequence of V0, V1, R, O steps, to be passed to LangevinSplittingIntegrator
    """
    assert(base_integrator == "VRORV" or base_integrator == "BAOAB")
    Rs = "R " * K_r
    inner_loop = "V1 " + Rs + "O " + Rs + "V1 "
    s = "V0 " + inner_loop * K_p + "V0"
    return s

def configure_platform(platform_name='Reference'):
    """Set precision, etc..."""
    if platform_name.upper() == 'Reference'.upper():
        platform = mm.Platform.getPlatformByName('Reference')
    elif platform_name.upper() == 'OpenCL'.upper():
        platform = mm.Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue('OpenCLPrecision', 'mixed')
    elif platform_name.upper() == 'CUDA'.upper():
        platform = mm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('CUDAPrecision', 'mixed')
    else:
        raise(ValueError("Invalid platform name"))
    return platform

def strip_unit(quantity):
    """Take a unit'd quantity and return just its value."""
    return quantity.value_in_unit(quantity.unit)

def load_waterbox(constrained=True):
    """Load WaterBox test system with non-default PME cutoff and error tolerance... """
    testsystem = WaterBox(constrained=constrained, ewaldErrorTolerance=1e-5, cutoff=10*unit.angstroms)
    (topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions
    positions = np.load("waterbox.npy") # load equilibrated configuration saved beforehand
    return topology, system, positions

def load_alanine(constrained=True):
    if constrained: constraints = app.HBonds
    else: constraints = None
    testsystem = AlanineDipeptideVacuum(constraints=constraints)
    topology, system, positions = testsystem.topology, testsystem.system, testsystem.positions

    index_of_CMMotionRemover = -1
    for i in range(system.getNumForces()):
        if type(system.getForce(i)) == mm.CMMotionRemover:
            index_of_CMMotionRemover = i
    if index_of_CMMotionRemover != -1:
        system.removeForce(i)

    return topology, system, positions

def get_total_energy(simulation):
    """Compute the kinetic energy + potential energy of the simulation."""
    state = simulation.context.getState(getEnergy=True)
    ke, pe = state.getKineticEnergy(), state.getPotentialEnergy()
    return ke + pe

def stderr(array):
    """Compute the standard error of an array."""
    return np.std(array) / np.sqrt(len(array))

def summarize(array):
    """Given an array, return a string with mean +/- 1.96 * standard error"""
    return "{:.3f} +/- {:.3f}".format(np.mean(array), 1.96 * stderr(array))

def get_summary_string(result, linebreaks=True):
    """Unpack a "result" tuple and return a summary string."""
    W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoint = result
    if linebreaks: separator = "\n\t"
    else: separator = ", "

    summary_string = separator.join(
        ["DeltaF_neq = {:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(sq_uncertainty)),
        "<W_F> = {}".format(summarize(np.array(W_shads_F)[:, -1])),
        "<W_midpoint> = {}".format(summarize(W_midpoint)),
        "<W_R> = {}".format(summarize(np.array(W_shads_R)[:, -1]))])
    return summary_string

def unpack_trajs(result):
    """Unpack a result tuple into x and y coordinates suitable for plotting."""
    W_shads_F, W_shads_R, DeltaF_neq, sq_uncertainty, W_midpoint = result
    Fs = []
    Rs = []

    for i in range(len(W_shads_F)):
        traj_F = W_shads_F[i]
        traj_R = W_shads_R[i] + traj_F[-1] + W_midpoint[i]

        x_F = (np.arange(len(traj_F)))
        x_R = (np.arange(len(traj_R))) + x_F[-1]

        Fs.append(traj_F)
        Rs.append(traj_R)

    return x_F, x_R, Fs, Rs

def plot(results, name=""):
    """Given a results dictionary that maps scheme strings to work trajectories / DeltaFNeq estimates, plot
    the trajectories and their averages, and print a summary."""
    schemes = results.keys()

    # get min and max
    y_min, y_max = np.inf, -np.inf
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])
        y_min_ = min(np.min(Fs), np.min(Rs))
        y_max_ = max(np.max(Fs), np.max(Rs))
        if y_min_ < y_min: y_min = y_min_
        if y_max_ > y_max: y_max = y_max_

    # plot individuals
    for scheme in schemes:

        # plot the raw shadow work trajectories
        plt.figure()
        plt.title(scheme + get_summary_string(results[scheme], linebreaks=False))

        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        traj_style = {"linewidth": 0.1, "alpha": 0.3}
        for i in range(len(Fs)):
            plt.plot(x_F, Fs[i], color='blue', **traj_style)
            plt.plot(x_R, Rs[i], color='green', **traj_style)

        plt.ylim(y_min, y_max)
        plt.xlabel('# steps')
        plt.ylabel('Shadow work')
        savefig('{}_work_trajectories_{}'.format(name, scheme))

        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color="blue")

        plt.plot(x_R, R_mean, color="green")

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color="blue", alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color="green", alpha=0.3)

        savefig('{}_averaged_work_trajectories_{}'.format(name, scheme))
        plt.close()
    # also make a comparison figure with all of the integrators, just with the confidence bands
    # instead of the full trajectories
    colors = dict(zip(schemes, "blue green orange purple darkviolet".split()))
    plt.figure()
    for scheme in schemes:
        x_F, x_R, Fs, Rs = unpack_trajs(results[scheme])

        # plot the averages
        F_mean = np.mean(Fs, 0)
        F_band = 1.96 * np.std(Fs, 0) / np.sqrt(len(Fs))

        R_mean = np.mean(Rs, 0)
        R_band = 1.96 * np.std(Rs, 0) / np.sqrt(len(Rs))

        plt.plot(x_F, F_mean, color=colors[scheme], label=scheme)
        plt.plot(x_R, R_mean, color=colors[scheme])

        # also shade in +/- 95% confidence limit...
        plt.fill_between(x_F, F_mean - F_band, F_mean + F_band, color=colors[scheme], alpha=0.3)
        plt.fill_between(x_R, R_mean - R_band, R_mean + R_band, color=colors[scheme], alpha=0.3)

        # plot vertical line where midpoint operator is applied
        # plt.vlines((x_F[-1] + x_R[0])/2.0, y_min, y_max, linestyles='--', color='grey')

    plt.xlabel('# steps')
    plt.ylabel('Shadow work')
    plt.title('Comparison')
    plt.legend(fancybox=True, loc='best')
    savefig('{}_averaged_work_trajectories_comparison'.format(name))
    plt.close()

def measure_shadow_work_via_W_shad(simulation, n_steps):
    """Simulate for n_steps, and record the integrator's W_shad global variable
    at each step, minus the value of W_shad before integrating."""
    get_W_shad = lambda : simulation.integrator.getGlobalVariableByName("W_shad")
    W_shads = np.zeros(n_steps)
    init_W_shad = get_W_shad()
    for i in range(n_steps):
        simulation.step(1)
        W_shads[i] = get_W_shad()
    return W_shads - init_W_shad

def measure_shadow_work_via_heat(simulation, n_steps):
    """Given a `simulation` that uses an integrator that accumulates heat exchange with bath,
    apply the integrator for n_steps and return the change in energy - the heat."""
    get_energy = lambda : get_total_energy(simulation)
    get_heat = lambda : simulation.integrator.getGlobalVariableByName("heat")

    E_0 = get_energy()
    Q_0 = get_heat()

    W_shads = []

    for _ in range(n_steps):
        simulation.step(1)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        W_shad = delta_E.value_in_unit(W_unit) - delta_Q
        W_shads.append(W_shad)

    return np.array(W_shads)

def measure_shadow_work_comparison(simulation, n_steps):
    """Measure shadow work using the global W_shad, and as DeltaE - heat, and raise
    a RuntimeWarning if they are inconsistent."""
    get_energy = lambda: get_total_energy(simulation)
    get_heat = lambda: simulation.integrator.getGlobalVariableByName("heat")
    get_W_shad = lambda: simulation.integrator.getGlobalVariableByName("W_shad")

    E_0 = get_energy()
    Q_0 = get_heat()
    init_W_shad = get_W_shad()

    W_shads_direct = np.zeros(n_steps)
    W_shads_Q = np.zeros(n_steps)

    for i in range(n_steps):
        simulation.step(1)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        W_shads_Q[i] = delta_E.value_in_unit(W_unit) - delta_Q
        W_shads_direct[i] = get_W_shad() - init_W_shad

    if np.linalg.norm(W_shads_direct - W_shads_Q) > 1e-8:
        raise (RuntimeWarning("Two methods of measuring shadow work were inconsistent!"))

    return W_shads_Q

def measure_shadow_work(simulation, n_steps):
    """Run the simulation for n_steps and return a vector of the shadow work accumulated
    during integration.

    * Check whether simulation.integrator has bookkeeping variables W_shad and/or heat.
    * If only W_shad is available, measure shadow work as W_shad
    * If only heat is available, measure shadow work as DeltaE - heat
    * If both are available, measure shadow work both ways and check for consistency.
    * If nether are available, raise a RuntimeError."""

    global_variable_names = [simulation.integrator.getGlobalVariableName(i) for i in range(simulation.integrator.getNumGlobalVariables())]

    if ("heat" in global_variable_names) and ("W_shad" in global_variable_names):
        return measure_shadow_work_comparison(simulation, n_steps)
    elif ("heat" in global_variable_names):
        return measure_shadow_work_via_heat(simulation, n_steps)
    elif ("W_shad" in global_variable_names):
        return measure_shadow_work_via_W_shad(simulation, n_steps)
    else:
        raise (RuntimeError("Simulation doesn't support shadow work computation"))

if __name__=="__main__":
    topology, system, positions = load_alanine(False)
    print(system.getForces())

