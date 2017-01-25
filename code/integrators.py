import numpy
import simtk.unit
import simtk.openmm as mm
print('OpenMM version: ', mm.version.full_version)
from openmmtools.constants import kB

class LangevinSplittingIntegrator(mm.CustomIntegrator):
    """Integrates Langevin dynamics with a prescribed operator splitting.

    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift"
            Deterministic update of *positions*, using current velocities
        - V: Linear "kick"
            Deterministic update of *velocities*, using current forces
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath

    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive by slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)

    Examples
    --------
        - VVVR:
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_p=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 V1 R R O R R V1 V0"
    """

    def __init__(self,
                 splitting="V R O R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=91.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 override_splitting_checks=False,
                 measure_shadow_work=True,
                 measure_heat=True
                 ):
        """

        Parameters
        ----------
        splitting : string
            Sequence of R, V, O substeps to be executed each timestep.

            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.

        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        measure_shadow_work : boolean
            Accumulate the shadow work performed by the symplectic substeps

        measure_heat : boolean
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        """

        # Compute constants
        kT = kB * temperature
        gamma = collision_rate
        kinetic_energy = "0.5 * m * v * v"

        splitting = splitting.upper().split()
        if override_splitting_checks == False:
            assert (set(splitting) == set("RVO"))

        # Count how many times each step appears, so we know how big each R/V/O substep will be
        n_R = sum([letter == "R" for letter in splitting])
        n_V = sum([letter == "V" for letter in splitting])
        n_O = sum([letter == "O" for letter in splitting])
        # Each force group should be integrated for a total length equal to dt
        fgs = set([step for step in splitting if step[0] == 'V'])
        n_Vs = dict()
        for fg in fgs: n_Vs[fg] = sum([step == fg for step in splitting])

        # Define substep functions
        def R_step():
            if measure_shadow_work:
                # store pre-step energy
                self.addComputeSum("old_ke", kinetic_energy)
                self.addComputeGlobal("old_pe", "energy")

            # update positions (and velocities, if there are constraints)
            self.addComputePerDof("x", "x + ((dt / n_R) * v)")
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / n_R))")
            self.addConstrainVelocities()

            if measure_shadow_work:
                # store post-step energy
                self.addComputeGlobal("new_pe", "energy")
                self.addComputeSum("new_ke", kinetic_energy)

                # accumulate shadow work
                self.addComputeGlobal("W_shad", "W_shad + ((new_ke + new_pe) - (old_ke - old_pe))")

        def V_step(fg):
            if measure_shadow_work:
                # store pre-step energy
                self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            self.addUpdateContextState()
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(n_Vs[fg], fg))
            self.addConstrainVelocities()
            self.addUpdateContextState()
            
            if measure_shadow_work:
                # store post-step energy
                self.addComputeSum("new_ke", kinetic_energy)

                # accumulate shadow work
                self.addComputeGlobal("W_shad", "W_shad + (new_ke - old_ke)")

        def O_step():
            if measure_heat:
                # store pre-step energy
                self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            self.addComputePerDof("v", "(b * v) + sqrt(1 - b*b) * sqrt(kT / m) * gaussian")
            self.addConstrainVelocities()

            if measure_heat:
                # store post-step energy
                self.addComputeSum("new_ke", kinetic_energy)

                # accumulate heat
                self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

        substep_functions = {"O": O_step, "R": R_step, "V": V_step }

        # Create a new CustomIntegrator
        super(LangevinSplittingIntegrator, self).__init__(timestep)
        # Initialize
        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma * timestep / max(1, n_O)))  # velocity mixing parameter
        self.addPerDofVariable('x1', 0) # positions before application of position constraints
        self.setConstraintTolerance(constraint_tolerance)

        self.addGlobalVariable("n_R", n_R)
        self.addGlobalVariable("n_V", n_V)
        self.addGlobalVariable("n_O", n_O)

        # Add bookkeeping variables
        if measure_heat or measure_shadow_work:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)
        if measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("W_shad", 0)
        if measure_heat:
            self.addGlobalVariable("heat", 0)

        # Integrate, applying constraints or bookkeeping as necessary
        self.addUpdateContextState()
        for step in splitting: substep_functions[step]()