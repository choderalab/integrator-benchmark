def keep_only_some_forces(system, extra_forces_to_keep=[]):
    """Remove unwanted forces, e.g. center-of-mass motion removal"""
    forces_to_keep = extra_forces_to_keep + ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce"]
    force_indices_to_remove = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if force.__class__.__name__ not in forces_to_keep:
            force_indices_to_remove.append(force_index)
    for force_index in force_indices_to_remove[::-1]:
        print('   Removing %s' % system.getForce(force_index).__class__.__name__)
        system.removeForce(force_index)