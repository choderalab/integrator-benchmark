import numpy
import simtk.unit
import simtk.openmm as mm

print('OpenMM version: ', mm.version.full_version)
from openmmtools.constants import kB


class CustomizableGHMC(mm.CustomIntegrator):
    """Customizable GHMC: proposal can contain mix of deterministic and stochastic steps.
    If the provided `splitting` is deterministic, then we add the stochastic velocity update to the beginning of the splitting.

    The class of allowable splittings is the same as for the LangevinSplittingIntegrator
    """

    def __init__(self,
                 splitting="V R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=91.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-4,
                 ):
        """
        Parameters
        ----------
        splitting : string
            Sequence of R, V substeps to be executed each timestep.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.

        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Temperature of heat bath

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        constraint_tolerance : float
            Numerical tolerance for solving constraints
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

        # If the splitting is deterministic, add a velocity-randomization step to the beginning
        # of the procedure
        if n_O == 0:
            splitting = ["O"] + splitting

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
            # update positions (and velocities, if there are constraints)
            self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
            self.addConstrainVelocities()

        def V_step(fg):
            """Deterministic velocity update, using only forces from force-group fg.
            Parameters
            ----------
            fg : string
                Force group to use in this substep.
                "" means all forces, "0" means force-group 0, etc.
            """
            # update velocities
            if mts:
                self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(n_Vs[fg], fg))
            else:
                self.addComputePerDof("v", "v + (dt / {}) * f / m".format(n_V))
            self.addConstrainVelocities()

        def O_step():
            # measure heat
            self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            self.addComputePerDof("v", "(a * v) + (b * sqrt(kT / m) * gaussian)")
            self.addConstrainVelocities()

            # measure heat
            self.addComputeSum("new_ke", kinetic_energy)
            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

        def substep_function(step_string):
            if step_string == "R":
                R_step()
            elif step_string[0] == "V":
                V_step(step_string[1:])

        def compute_total_energy(name="e_old"):
            self.addComputeSum("ke", kinetic_energy)
            self.addComputeGlobal(name, "ke + energy")

        # Create a new CustomIntegrator
        super(CustomizableGHMC, self).__init__(timestep)

        # Initialize
        self.addGlobalVariable("kT", kT)

        # Velocity mixing parameter: current velocity component
        h = timestep
        self.addGlobalVariable("a", numpy.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", numpy.sqrt(1 - numpy.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add bookkeeping variables
        self.addGlobalVariable("ke_old", 0)
        self.addGlobalVariable("ke_new", 0)
        self.addGlobalVariable("heat", 0)
        self.addGlobalVariable("W_shad", 0)
        self.addGlobalVariable("e_old", 0)
        self.addGlobalVariable("e_new", 0)
        self.addGlobalVariable("acc_ratio", 0)
        self.addGlobalVariable("accept", 0)
        self.addGlobalVariable("naccept", 0)
        self.addGlobalVariable("ntrials", 0)
        self.addPerDofVariable("xold", 0)
        self.addPerDofVariable("vold", 0)

        # Compute energy of current state
        self.addUpdateContextState()
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        compute_total_energy("e_old")

        # Reset "heat" to zero
        self.addComputeGlobal("heat", "0")

        # Integrate / generate proposal
        for i, step in enumerate(splitting):
            substep_function(step)

        # Compute M-H ratio in terms of energy change during the deterministic steps
        compute_total_energy("e_new")
        self.addComputeGlobal("W_shad", "(e_new - e_old) - heat")
        self.addComputeGlobal("acc_ratio", "exp(- W_shad / kT)")

        # Accept / reject : flip momenta upon rejection
        self.addComputeGlobal("accept", "step(acc_ratio - uniform)")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.endBlock()

        # Update acceptance statistics
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")
