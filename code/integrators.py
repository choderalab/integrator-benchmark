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
            x <- x + v dt

        - V: Linear "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass

        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal

    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
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
            assert (set(splitting).issubset(set("RVO").union(set(["V{}".format(i) for i in range(32)]))))
            assert ("R" in splitting)
            assert ("O" in splitting)

        # Count how many times each step appears, so we know how big each R/V/O substep will be
        n_R = sum([letter == "R" for letter in splitting])
        n_V = sum([letter == "V" for letter in splitting])
        n_O = sum([letter == "O" for letter in splitting])
        # Each force group should be integrated for a total length equal to dt
        fgs = set([step[1:] for step in splitting if step[0] == 'V'])
        n_Vs = dict()
        for fg in fgs: n_Vs[fg] = sum([step[1:] == fg for step in splitting])

        def get_total_energy(name="old"):
            """Computes
            * kinetic energy in global ke
            * potential energy in global pe
            * total energy in global {name}_e
            """
            self.addComputeSum("ke", kinetic_energy)
            self.addComputeGlobal("pe", "energy")
            self.addComputeGlobal("{}_e".format(name), "pe + ke")

        def compute_substep_energy_change():
            get_total_energy("new")
            self.addComputeGlobal("current_DeltaE", "new_e - old_e")
            self.addComputeGlobal("old_e", "new_e")

        # Define substep functions
        def R_step():
            # update positions (and velocities, if there are constraints)
            self.addComputePerDof("x", "x + ((dt / n_R) * v)")
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / n_R))")
            self.addConstrainVelocities()

            compute_substep_energy_change()
            if measure_shadow_work: self.addComputeGlobal("W_shad", "W_shad + current_DeltaE")

        def V_step(fg):
            """Deterministic velocity update, using only forces from force-group fg.

            Parameters
            ----------
            fg : string
                Force group to use in this substep.
                "" means all forces, "0" means force-group 0, etc.
            """
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(n_Vs[fg], fg))
            self.addConstrainVelocities()
            self.addUpdateContextState()

            compute_substep_energy_change()
            if measure_shadow_work: self.addComputeGlobal("W_shad", "W_shad + current_DeltaE")

        def O_step():
            # update velocities
            self.addComputePerDof("v", "(a * v) + (b * sqrt(kT / m) * gaussian)")
            self.addConstrainVelocities()

            compute_substep_energy_change()
            if measure_heat: self.addComputeGlobal("heat", "heat + current_DeltaE")

        def substep_function(step_string):
            if step_string == "O":
                O_step()
            elif step_string == "R":
                R_step()
            elif step_string[0] == "V":
                V_step(step_string[1:])

        # Create a new CustomIntegrator
        super(LangevinSplittingIntegrator, self).__init__(timestep)
        # Initialize
        self.addGlobalVariable("kT", kT)  # thermal energy

        # velocity mixing parameter: current velocity
        h = timestep / max(1, n_O)
        self.addGlobalVariable("a", numpy.exp(-gamma * h))

        # velocity mixing parameter: random velocity
        self.addGlobalVariable("b", numpy.sqrt(1 - numpy.exp(- 2 * gamma * h)))
        self.addPerDofVariable("x1", 0) # positions before application of position constraints

        # bookkeeping variables
        self.addPerDofVariable("x_prev", 0)
        self.addPerDofVariable("v_prev", 0)

        self.setConstraintTolerance(constraint_tolerance)

        self.addGlobalVariable("n_R", n_R)
        self.addGlobalVariable("n_V", n_V)
        self.addGlobalVariable("n_O", n_O)

        # Add bookkeeping variables
        self.addGlobalVariable("ke", 0)
        self.addGlobalVariable("pe", 0)
        self.addGlobalVariable("old_e", 0)
        self.addGlobalVariable("new_e", 0)

        self.addGlobalVariable("W_shad", 0)
        self.addGlobalVariable("heat", 0)

        # store all of the substep energy changes
        self.addGlobalVariable("current_DeltaE", 0)
        for i in range(len(splitting)):
            self.addGlobalVariable("DeltaE_{}".format(i), 0)

        # Integrate, applying constraints or bookkeeping as necessary
        get_total_energy("old")
        self.addUpdateContextState()

        # measure energy change in each substep...
        for i, step in enumerate(splitting):
            substep_function(step)
            self.addComputeGlobal("DeltaE_{}".format(i), "current_DeltaE")