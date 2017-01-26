from simtk import unit
from simtk.openmm import app
import numpy as np
from integrators import LangevinSplittingIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
W_unit = unit.kilojoule_per_mole
from utils import configure_platform, get_total_energy, strip_unit, generate_solvent_solute_splitting_string, load_alanine


def get_n_substeps(integrator):
    matching_string = "DeltaE_"
    global_var_names = [integrator.getGlobalVariableName(i) for i in range(integrator.getNumGlobalVariables())]
    n_substeps = len([var_name for var_name in global_var_names if var_name[:len(matching_string)] == matching_string])
    return n_substeps

def get_substep_energy_changes(integrator, n_substeps):
    DeltaEs = np.zeros(n_substeps)
    for i in range(n_substeps):
        DeltaEs[i] = integrator.getGlobalVariableByName("DeltaE_{}".format(i))
    return DeltaEs

def compare_substep_energy_changes(simulation, n_steps=10):
    """Get all the substep energy changes, and compare them with
    """
    n_substeps = get_n_substeps(simulation.integrator)
    substep_DeltaEs = np.zeros((n_steps, n_substeps))
    onstep_DeltaEs = np.zeros(n_steps)
    get_energy = lambda: get_total_energy(simulation)


    for i in range(n_steps):
        E_0 = get_energy()
        simulation.step(1)
        E_1 = get_energy()

        onstep_DeltaEs[i] = strip_unit(E_1 - E_0)
        substep_DeltaEs[i] = get_substep_energy_changes(simulation.integrator, n_substeps)

    deviations = onstep_DeltaEs - substep_DeltaEs.sum(1)
    print("Deviation in first step: {:.3f}".format(deviations[0]))
    print("Deviations in subsequent steps: {}".format(deviations[1:]))

    return onstep_DeltaEs, substep_DeltaEs

def record_energy_changes(simulation, n_steps=100, W_shad_name="W_shad"):
    """Record the per-step changes in
    total energy,
    integrator-accumulated "heat,"
    and integrator-accumulated "W_shad"
    for n_steps."""

    get_energy = lambda: get_total_energy(simulation)
    get_heat = lambda: simulation.integrator.getGlobalVariableByName("heat")
    get_W_shad = lambda: simulation.integrator.getGlobalVariableByName(W_shad_name)

    delta_energy = np.zeros(n_steps)
    delta_heat = np.zeros(n_steps)
    delta_W_shad = np.zeros(n_steps)

    for i in range(n_steps):
        e_0 = get_energy()
        q_0 =  get_heat()
        W_shad_0 = get_W_shad()

        simulation.step(1)

        delta_energy[i] = strip_unit(get_energy() - e_0)
        delta_heat[i] = get_heat() - q_0
        delta_W_shad[i] = get_W_shad() - W_shad_0


    return delta_energy, delta_heat, delta_W_shad

def simulation_factory(scheme, constrained=True):
    """Create and return a simulation that includes:
    * Langevin integrator with the prescribed operator splitting
    * AlanineDipeptideVacuum with or without restraints."""
    platform = configure_platform("Reference")
    temperature = 298 * unit.kelvin
    topology, system, positions = load_alanine(constrained)

    lsi = LangevinSplittingIntegrator(scheme, temperature=temperature,
                                      measure_heat=True, measure_shadow_work=True)

    simulation = app.Simulation(topology, system, lsi, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation

def check_W_shad_consistency(energy_changes):
    """Compute and print the average deviation from our definition that
    Delta E = W_shad + Q for each timestep."""
    delta_energy, delta_heat, delta_W_shad = energy_changes
    print("\tAverage deviation between Delta E and (W_shad + Q): {:.3f}".format(
        np.linalg.norm(delta_energy - (delta_W_shad + delta_heat)) / len(delta_energy)))


if __name__ == "__main__":
    # loop over several operator splittings, with and without constraints
    for constrained in [True, False]:
        if constrained: print("\n\nWith constraints\n")
        else: print("\n\nWithout constraints\n")

        for scheme in ["R V O", "R O V",
                       "V R O R V", "R V O V R", "O R V R O",
                       "R O V O R", "V O R O V", "V R R R O R R R V",
                       generate_solvent_solute_splitting_string(K_p=2,K_r=2)
                       ]:
            simulation = simulation_factory(scheme)
            #simulation.step(100)
            print("Scheme: {}".format(scheme))
            check_W_shad_consistency(record_energy_changes(simulation))

    # also check VVVR, in the system without constraints
    print("\n\nVVVR")
    platform = configure_platform("Reference")
    temperature = 298 * unit.kelvin
    testsystem = AlanineDipeptideVacuum(constraints=None)
    from openmmtools.integrators import VVVRIntegrator
    vvvr = VVVRIntegrator(temperature, monitor_heat=True, monitor_work=True)
    simulation = app.Simulation(testsystem.topology, testsystem.system, vvvr, platform)
    simulation.context.setPositions(testsystem.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(100)
    check_W_shad_consistency(record_energy_changes(simulation, W_shad_name="shadow_work"))