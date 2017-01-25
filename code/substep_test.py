from tests import compare_substep_energy_changes, simulation_factory
from utils import generate_solvent_solute_splitting_string

for constrained in [True, False]:
    if constrained: print("\n\nWith constraints\n")
    else: print("\n\nWithout constraints\n")

    for scheme in ["R V O", "R O V",
                   "V R O R V", "R V O V R", "O R V R O",
                   "R O V O R", "V O R O V", "V R R R O R R R V",
                   generate_solvent_solute_splitting_string(K_p=2,K_r=2)
                   ]:
        simulation = simulation_factory(scheme)
        print("Scheme: {}".format(scheme))
        compare_substep_energy_changes(simulation, n_steps=10)