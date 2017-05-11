import numpy as np
from benchmark.integrators import LangevinSplittingIntegrator
from benchmark.utilities import get_total_energy
from simtk import unit
from simtk.openmm import app

W_unit = unit.kilojoule_per_mole
from benchmark.testsystems.configuration import configure_platform
from benchmark.testsystems import system_loaders
from benchmark import simulation_parameters


def simulation_factory(scheme, system_loader, constrained=True):
    """Create and return a simulation that includes:
    * Langevin integrator with the prescribed operator splitting
    * AlanineDipeptideVacuum with or without restraints."""
    platform = configure_platform("Reference")
    temperature = simulation_parameters["temperature"]
    topology, system, positions = system_loader(constrained)

    lsi = LangevinSplittingIntegrator(scheme, temperature=temperature, timestep=2.0 * unit.femtosecond,
                                      measure_heat=True, measure_shadow_work=True)

    simulation = app.Simulation(topology, system, lsi, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation


def check_W_shad_consistency(simulation, system_name="", n_steps=100, threshold=1e-6):
    """Compute and print the average deviation from our definition that
    Delta E = W_shad + Q for each timestep."""
    W_shads_Q, W_shads_direct = measure_shadow_work_comparison(simulation, n_steps, return_both=True)

    deviation = np.linalg.norm(W_shads_Q - W_shads_direct)
    if deviation > threshold:
        message = "Discrepancy between two methods for measuring shadow work" \
                  "exceeds threshold (})!".format(threshold)
        if len(system_name) > 0:
            message = message + "\n\t{}".format(system_name)
        message = message + "\n\tDeviation between two methods for computing shadow work: {:.3f}".format(deviation)
        raise (RuntimeError(message))


def test_W_shad_consistency():
    # loop over several operator splittings, with and without constraints
    system_loader = system_loaders["alanine"]
    for constrained in [True, False]:
        for scheme in ["R V O", "R O V",
                       "V R O R V", "R V O V R", "O R V R O",
                       "R O V O R", "V O R O V", "V R R O R R V"]:
            simulation = simulation_factory(scheme, system_loader, constrained)
            simulation.step(10)
            system_name = "Scheme: {} Constrained: {}".format(scheme, constrained)
            yield check_W_shad_consistency, simulation, system_name


def measure_shadow_work_directly(simulation, n_steps):
    """Simulate for n_steps, and record the integrator's shadow_work global variable
    at each step, minus the value of shadow_work before integrating."""
    get_shadow_work = lambda: simulation.integrator.getGlobalVariableByName("shadow_work")
    shadow_work = np.zeros(n_steps)
    init_shadow_work = get_shadow_work()
    for i in range(n_steps):
        simulation.step(1)
        shadow_work[i] = get_shadow_work()
    return shadow_work - init_shadow_work


def measure_shadow_work_via_heat(simulation, n_steps):
    """Given a `simulation` that uses an integrator that accumulates heat exchange with bath,
    apply the integrator for n_steps and return the change in energy - the heat."""
    get_energy = lambda: get_total_energy(simulation)
    get_heat = lambda: simulation.integrator.getGlobalVariableByName("heat")

    E_0 = get_energy()
    Q_0 = get_heat()

    shadow_work = np.zeros(n_steps)

    for i in range(n_steps):
        simulation.step(1)

        E_1 = get_energy()
        Q_1 = get_heat()

        delta_E = E_1 - E_0
        delta_Q = Q_1 - Q_0

        shadow_work[i] = delta_E.value_in_unit(W_unit) - delta_Q

    return shadow_work


def measure_shadow_work_comparison(simulation, n_steps, return_both=False):
    """Measure shadow work using the global shadow_work, and as DeltaE - heat, and raise
    a RuntimeWarning if they are inconsistent."""
    get_energy = lambda: get_total_energy(simulation)
    get_heat = lambda: simulation.integrator.getGlobalVariableByName("heat")
    get_W_shad = lambda: simulation.integrator.getGlobalVariableByName("shadow_work")

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

    if np.linalg.norm(W_shads_direct - W_shads_Q) > 1e-6:
        raise (RuntimeWarning("Two methods of measuring shadow work were inconsistent!"))

    if return_both:
        return W_shads_Q, W_shads_direct
    else:
        return W_shads_Q


def measure_shadow_work(simulation, n_steps):
    """Run the simulation for n_steps and return a vector of the shadow work accumulated
    during integration.

    * Check whether simulation.integrator has bookkeeping variables W_shad and/or heat.
    * If only W_shad is available, measure shadow work as W_shad
    * If only heat is available, measure shadow work as DeltaE - heat
    * If both are available, measure shadow work both ways and check for consistency.
    * If nether are available, raise a RuntimeError."""

    global_variable_names = [simulation.integrator.getGlobalVariableName(i) for i in
                             range(simulation.integrator.getNumGlobalVariables())]

    if ("heat" in global_variable_names) and ("shadow_work" in global_variable_names):
        return measure_shadow_work_comparison(simulation, n_steps)
    elif ("heat" in global_variable_names):
        return measure_shadow_work_via_heat(simulation, n_steps)
    elif ("shadow_work" in global_variable_names):
        return measure_shadow_work_directly(simulation, n_steps)
    else:
        raise (RuntimeError("Simulation doesn't support shadow work computation"))
