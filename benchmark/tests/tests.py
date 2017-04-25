import numpy as np
from benchmark.integrators import LangevinSplittingIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit
from simtk.openmm import app

W_unit = unit.kilojoule_per_mole
from benchmark.testsystems.configuration import configure_platform
from benchmark.integrators import generate_solvent_solute_splitting_string
from benchmark.testsystems.testsystems import load_alanine
from measure_shadow_work import measure_shadow_work_comparison

def simulation_factory(scheme, constrained=True):
    """Create and return a simulation that includes:
    * Langevin integrator with the prescribed operator splitting
    * AlanineDipeptideVacuum with or without restraints."""
    platform = configure_platform("Reference")
    temperature = 298 * unit.kelvin
    topology, system, positions = load_alanine(constrained)

    lsi = LangevinSplittingIntegrator(scheme, temperature=temperature, timestep=2.0*unit.femtosecond,
                                      measure_heat=True, measure_shadow_work=True)

    simulation = app.Simulation(topology, system, lsi, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation

def check_W_shad_consistency(simulation, n_steps=100, threshold=1e-8):
    """Compute and print the average deviation from our definition that
    Delta E = W_shad + Q for each timestep."""
    W_shads_Q, W_shads_direct = measure_shadow_work_comparison(simulation, n_steps, return_both=True)

    deviation = np.linalg.norm(W_shads_Q - W_shads_direct)
    print("\tDeviation between two methods for computing shadow work: {:.3f}".format(deviation))

    if deviation > threshold:
        raise(RuntimeError("Discrepancy between two methods for measuring shadow work "
                           "exceeds threshold!"))


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
            simulation.step(10)
            print("Scheme: {}".format(scheme))
            check_W_shad_consistency(simulation)
