# Instead of only considering substep lengths of dt or dt/2, why not continuously
# parameterize them?

import os

import numpy
import numpy as np
import simtk.openmm as mm
import simtk.unit
from simtk import unit

from benchmark import DATA_PATH
from benchmark.evaluation import estimate_nonequilibrium_free_energy
from benchmark.experiments.driver import ExperimentDescriptor, Experiment
from benchmark.testsystems import dhfr_constrained, NonequilibriumSimulator

print('OpenMM version: ', mm.version.full_version)
from openmmtools.constants import kB


class ContinuousLangevinSplittingIntegrator(mm.CustomIntegrator):
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
                 splitting=[("V", 0.5), ("R", 0.5), ("O", 1.0), ("R", 0.5), ("V", 0.5)],
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
        # splitting = splitting.upper().split()

        # Count how many times each step appears, so we know how big each R/V/O substep will be
        # n_R = sum([letter == "R" for letter in splitting])
        # n_V = sum([letter == "V" for letter in splitting])
        # n_O = sum([letter == "O" for letter in splitting])

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
            assert ("R" in [s[0] for s in splitting])
            assert ("V" in [s[0][0] for s in splitting])
            assert ("O" in [s[0] for s in splitting])

            lengths = {}
            for (step, length) in splitting:
                if step in lengths:
                    lengths[step] += length
                else:
                    lengths[step] = length

            assert (np.isclose(lengths["R"], 1.0))
            assert (np.isclose(lengths["V"], 1.0))
            assert (np.isclose(lengths["O"], 1.0))

            # TODO: UPDATE TO ACCOMODATE MULTI-TIMESTEP SCHEMES HERE
            # Make sure it contains no invalid characters
            # assert (set(splitting).issubset(set("RVO").union(set(["V{}".format(i) for i in range(32)]))))

            # If the splitting string contains both "V" and a force-group-specific V0,V1,etc.,
            # then raise an error
            # if mts and (n_V > 0):
            #    raise (ValueError("Splitting string includes an evaluation of all forces and "
            #                      "evaluation of subsets of forces."))

        # Define substep functions
        def R_step(length):
            """Length between 0 and 1"""
            if measure_shadow_work:
                self.addComputeGlobal("old_pe", "energy")
                self.addComputeSum("old_ke", kinetic_energy)

            # update positions (and velocities, if there are constraints)
            self.addComputePerDof("x", "x + ((dt * {}) * v)".format(length))
            self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
            self.addConstrainPositions()  # x is now constrained
            self.addComputePerDof("v", "v + ((x - x1) / (dt * {}))".format(length))
            self.addConstrainVelocities()

            if measure_shadow_work:
                self.addComputeGlobal("new_pe", "energy")
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

        def V_step(fg, length):
            """Deterministic velocity update, using only forces from force-group fg.

            Parameters
            ----------
            fg : string
                Force group to use in this substep.
                "" means all forces, "0" means force-group 0, etc.
            """
            if measure_shadow_work:
                self.addComputeSum("old_ke", kinetic_energy)

            # update velocities
            if mts:
                self.addComputePerDof("v", "v + ((dt * {}) * f{} / m)".format(length, fg))
            else:
                self.addComputePerDof("v", "v + (dt * {}) * f / m".format(length))

            self.addConstrainVelocities()

            if measure_shadow_work:
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

        def O_step(length):
            if measure_heat:
                self.addComputeSum("old_ke", kinetic_energy)

            h = timestep * length
            # update velocities
            self.addComputePerDof("v", "({} * v) + ({} * sqrt(kT / m) * gaussian)".format(
                numpy.exp(-gamma * h), numpy.sqrt(1 - numpy.exp(- 2 * gamma * h))))
            self.addConstrainVelocities()

            if measure_heat:
                self.addComputeSum("new_ke", kinetic_energy)
                self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

        def substep_function(step_string, length):
            if step_string == "O":
                O_step(length)
            elif step_string == "R":
                R_step(length)
            elif step_string[0] == "V":
                V_step(step_string[1:], length)

        # Create a new CustomIntegrator
        super(ContinuousLangevinSplittingIntegrator, self).__init__(timestep)

        # Initialize
        self.addGlobalVariable("kT", kT)

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
            self.addGlobalVariable("shadow_work", 0)

        # Integrate
        self.addUpdateContextState()
        for (step, length) in splitting:
            if length > 0:
                substep_function(step, length)


class CLSIExperiment(Experiment):
    def run(self):
        exp = self.experiment_descriptor
        simulator = NonequilibriumSimulator(exp.equilibrium_simulator,
                                            ContinuousLangevinSplittingIntegrator(
                                                splitting=exp.splitting_string,
                                                timestep=exp.timestep_in_fs * unit.femtosecond,
                                                collision_rate=exp.collision_rate))

        self.result = simulator.collect_protocol_samples(
            exp.n_protocol_samples, exp.protocol_length, exp.marginal,
            store_potential_energy_traces=(exp.marginal == "full" and self.store_potential_energy_traces))

        DeltaF_neq, squared_uncertainty = estimate_nonequilibrium_free_energy(self.result[0], self.result[1])
        print(self)
        print("\t{:.3f} +/- {:.3f}".format(DeltaF_neq, np.sqrt(squared_uncertainty)))


alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
splittings = {}

# Initial experiment:
# for alpha in alphas:
#    splittings["VRORV ({})".format(alpha)] = [
#        ("V", 0.5), ("R", alpha), ("O", 1.0), ("R", 1.0 - alpha), ("V", 0.5)
#    ]

# alpha = 0 -> VORV = RVO, rotated
# alpha = 1 -> VROV = VRO, rotated
# this was stupid, because:
# * at alpha=0, this is "hamiltonian independent"
# * we only recover a single "interesting method, at alpha=0.5
# * at alpha=0 or 1 we have a "first-order" method, so it's not a fair comparison...


# Update: let's interpolate between two interesting endpoints:
# Possibilities: OVRVO, VRORV, RVOVR, ORVRO
# How do I construct a continuous interpolation between two desired end-points?
# Let's say the desired endpoints are VRORV (alpha=1) and RVOVR (alpha=0)
# Could construct V_alpha/2 R V_(1-alpha)/2 O V_(1-alpha)/2 R V_alpha/2...
# for alpha in alphas:
#    splittings["VRVOVRV ({})".format(alpha)] = [
#        ("V", 0.5 * (alpha)),
#        ("R", 0.5),
#        ("V", 0.5 * (1 - alpha)),
#        ("O", 1.0),
#        ("V", 0.5 * (1 - alpha)),
#        ("R", 0.5),
#        ("V", 0.5 * (alpha)),
#    ]


# Final experimental design is two dimensional:
betas = np.array(alphas)
for alpha in alphas:
    for beta in betas:
        s = [
            ("O", 0.5 * (1 - beta)),
            ("V", 0.5 * (alpha)),
            ("R", 0.5),
            ("V", 0.5 * (1 - alpha)),
            ("O", 1.0 * beta),
            ("V", 0.5 * (1 - alpha)),
            ("R", 0.5),
            ("V", 0.5 * (alpha)),
            ("O", 0.5 * (1 - beta)),
        ]
        splittings["OVRVOVRVO ({}, {})".format(alpha, beta)] = s

# now, each (alpha, beta) corner is one of the splittings of interest:
# (0,0) : ORVRO
# (0,1) : RVOVR
# (1,0) : OVRVO
# (1,1) : VRORV

systems = {"DHFR in explicit solvent (constrained)": dhfr_constrained}

dt_range = np.array([3.0])  # chosen to be >0.5fs under OVRVO's stability limit for this system

marginals = ["configuration", "full"]

collision_rates = {"high": 91.0 / unit.picoseconds,
                   "low": 1.0 / unit.picoseconds
                   }

n_protocol_samples = 100
protocol_lengths = [2000]

experiment_name = "A2_continuous_lsi"
experiments = []
i = 1

for splitting_name in sorted(splittings.keys()):
    for system_name in sorted(systems.keys()):
        for dt in dt_range:
            for marginal in marginals:
                for collision_rate_name in sorted(collision_rates.keys()):
                    for protocol_length in protocol_lengths:
                        partial_fname = "{}_{}.pkl".format(experiment_name, i)
                        full_filename = os.path.join(DATA_PATH, partial_fname)

                        experiment_descriptor = ExperimentDescriptor(
                            experiment_name=experiment_name,
                            system_name=system_name,
                            equilibrium_simulator=systems[system_name],
                            splitting_name=splitting_name,
                            splitting_string=splittings[splitting_name],
                            timestep_in_fs=dt,
                            marginal=marginal,
                            collision_rate_name=collision_rate_name,
                            collision_rate=collision_rates[collision_rate_name],
                            n_protocol_samples=n_protocol_samples,
                            protocol_length=protocol_length,
                            h_mass_factor=1
                        )

                        print(i, splitting_name)

                        experiments.append(CLSIExperiment(experiment_descriptor, full_filename))
                        i += 1

if __name__ == "__main__":
    print(len(experiments))
    import sys

    job_id = int(sys.argv[1])
    experiments[job_id].run_and_save()
