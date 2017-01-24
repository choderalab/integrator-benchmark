import numpy
import simtk.unit
import simtk.openmm as mm
print('OpenMM version: ', mm.version.full_version)
from openmmtools.constants import kB

class VelocityConstrainer(mm.CustomIntegrator):
    def __init__(self, constraint_tolerance=1e-8):
        super(VelocityConstrainer, self).__init__(1.0 * simtk.unit.femtosecond)
        self.setConstraintTolerance(constraint_tolerance)

        self.addUpdateContextState()

        self.addComputePerDof("x", "x")
        self.addComputePerDof("v", "v")
        self.addConstrainVelocities()
        self.addConstrainVelocities()
        self.addConstrainVelocities()


class VelocityRandomizer(mm.CustomIntegrator):
    """Doesn't apply velocity constraints... """
    def __init__(self, temperature):
        super(VelocityRandomizer, self).__init__(1.0 * simtk.unit.femtosecond)
        self.addGlobalVariable("kT", kB * temperature)

        self.addUpdateContextState()
        self.addComputePerDof("x", "x")
        self.addComputePerDof("v", "sqrt(kT / m) * gaussian")

class FGLangevinSplittingIntegrator(mm.CustomIntegrator):
    '''Integrates Langevin dynamics with a prescribed splitting.

    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift"
            Deterministic update of *positions*, using current velocities
        - V: Linear "kick"
            Deterministic update of *velocities*, using current forces
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath

    We can then construct integrators by solving each part for a certain timestep in sequence.

    We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive by slow-fluctuating forces.

    Examples:
        - VVVR:
            "O V R V O"
        - BAOAB:
            "V R O R V"
        - g-BAOAB, with K_p=3:
            "V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            "V0 V1 R R O R R V1 V1 R R O R R V1 V0"
    '''

    def __init__(self,
                 splitting="V0 R O R V1",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=91.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8
                 ):
        '''
        Always monitor heat, since it costs essentially nothing...

        Accumulate the heat exchanged with the bath in each step, in the global `heat`

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
        '''
        # Compute constants
        kT = kB * temperature
        gamma = collision_rate

        splitting = splitting.upper().split()
        # there are at most 32 force groups in the system
        assert (set(splitting).issubset(set("RVO").union(set(["V{}".format(i) for i in range(32)]))))
        assert ("R" in splitting)
        assert ("O" in splitting)

        # Count how many times each step appears, so we know how big each R/V/O substep will be
        n_R = sum([letter == "R" for letter in splitting])
        n_O = sum([letter == "O" for letter in splitting])
        # Each force group should be integrated for a total length equal to dt
        fgs = set([step for step in splitting if step[0] == 'V'])
        n_Vs = dict()
        for fg in fgs: n_Vs[fg] = sum([step == fg for step in splitting])
        # To-do: add checks here.

        # Define substep functions
        def R_step():
            self.addComputePerDof("x", "x + (dt / {}) * v".format(n_R))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + (x - x1) / (dt / {})".format(n_R))
            self.addConstrainVelocities()

        def V_step(fg):
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(n_Vs[fg], fg))
            self.addConstrainVelocities()

        kinetic_energy = "0.5 * m * v * v"
        def O_step():
            self.addComputeSum("old_ke", kinetic_energy)

            self.addComputePerDof("v", "(b * v) + sqrt((1 - b*b) * (kT / m)) * gaussian")
            self.addConstrainVelocities()

            self.addComputeSum("new_ke", kinetic_energy)

            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

        def substep_function(step):
            ''' step is either O, R, V, or V{force_group}'''
            if step == "O": O_step()
            if step == "R": R_step()
            if step[0] == "V": V_step(step)

        # Create a new CustomIntegrator
        super(FGLangevinSplittingIntegrator, self).__init__(timestep)
        # Initialize
        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma * timestep / n_O))  # velocity mixing parameter
        self.addPerDofVariable('x1', 0) # positions before application of position constraints
        self.setConstraintTolerance(constraint_tolerance)

        # Add bookkeeping variables
        self.addGlobalVariable("old_ke", 0)
        self.addGlobalVariable("new_ke", 0)
        self.addGlobalVariable("heat", 0)

        # Integrate, applying constraints or bookkeeping as necessary
        for step in splitting: substep_function(step)