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
                 measure_shadow_work=False,
                 measure_heat=True,
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

        # Convert splitting string into a list of all-caps strings
        splitting = splitting.upper().split()

        # Count how many times each step appears, so we know how big each R/V/O substep will be
        n_R = sum([letter == "R" for letter in splitting])
        n_V = sum([letter == "V" for letter in splitting])
        n_O = sum([letter == "O" for letter in splitting])

        # Check if the splitting string asks for multi-time-stepping.
        # If so, each force group should be integrated for a total length equal to dt
        if len(set([step for step in splitting if step[0] == "V"])) > 1:
            mts = True
            fgs = set([step[1:] for step in splitting if step[0] == "V"])
            n_Vs = dict()
            for fg in fgs:
                n_Vs[fg] = sum([step[1:] == fg for step in splitting])
        else:
            mts = False

        # Do a couple sanity checks on the splitting string
        if override_splitting_checks == False:
            # Make sure we contain at least one of R, V, O steps
            assert ("R" in splitting)
            assert ("V" in [s[0] for s in splitting])
            assert ("O" in splitting)

            # Make sure it contains no invalid characters
            assert (set(splitting).issubset(set("RVO").union(set(["V{}".format(i) for i in range(32)]))))

            # If the splitting string contains both "V" and a force-group-specific V0,V1,etc.,
            # then raise an error
            if mts and (n_V > 0):
                raise (ValueError("Splitting string includes an evaluation of all forces and "
                                  "evaluation of subsets of forces."))

        # Define substep functions
        def R_step():
            self.addConstrainPositions()  # TODO: Constrain initial step only?
            self.addConstrainVelocities() # TODO: Constrain initial step only?

            if measure_shadow_work:
                self.addComputeGlobal("old_pe", "energy")
                self.addComputeSum("old_ke", kinetic_energy)

            # update positions (and velocities, if there are constraints)
            self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
            self.addConstrainVelocities()

            if measure_shadow_work:
                self.addComputeGlobal("new_pe", "energy")
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("W_shad", "W_shad + (new_ke + new_pe) - (old_ke + old_pe)")

        def V_step(fg):
            """Deterministic velocity update, using only forces from force-group fg.

            Parameters
            ----------
            fg : string
                Force group to use in this substep.
                "" means all forces, "0" means force-group 0, etc.
            """
            self.addConstrainPositions()  # TODO: Constrain initial step only?
            self.addConstrainVelocities() # TODO: Constrain initial step only?

            if measure_shadow_work:
                self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            if mts:
                self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(n_Vs[fg], fg))
            else:
                self.addComputePerDof("v", "v + (dt / {}) * f / m".format(n_V))

            self.addConstrainVelocities()

            if measure_shadow_work:
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

        def O_step():
            self.addConstrainPositions()  # TODO: Constrain initial step only?
            self.addConstrainVelocities() # TODO: Constrain initial step only?

            if measure_heat:
                self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            self.addComputePerDof("v", "(a * v) + (b * sqrt(kT / m) * gaussian)")
            self.addConstrainVelocities()

            if measure_heat:
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

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
        self.addGlobalVariable("kT", kT)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, n_O)
        self.addGlobalVariable("a", numpy.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", numpy.sqrt(1 - numpy.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add bookkeeping variables
        if measure_heat:
            self.addGlobalVariable("heat", 0)

        if measure_shadow_work or measure_heat:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)

        if measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("W_shad", 0)

        # Integrate
        self.addUpdateContextState()
        for i, step in enumerate(splitting):
            substep_function(step)
