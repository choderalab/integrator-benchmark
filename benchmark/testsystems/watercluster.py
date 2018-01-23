# TODO: Replace WaterCluster class definition with import from openmmtools, when released

from openmmtools.testsystems import *
from simtk import unit
from benchmark.testsystems.configuration import configure_platform
from benchmark import simulation_parameters
temperature = simulation_parameters["temperature"]
from benchmark.testsystems.bookkeepers import EquilibriumSimulator

class WaterCluster(TestSystem):
    """Create a few water molecules in a harmonic restraining potential"""

    def __init__(self,
                 n_waters=20,
                 K=1.0 * unit.kilojoules_per_mole / unit.nanometer ** 2,
                 model='tip3p',
                 constrained=True,
                 **kwargs):
        """
        Parameters
        ----------
        n_waters : int
            Number of water molecules in the cluster
        K : simtk.unit.Quantity (energy / distance^2)
            spring constant for restraining potential
        model : string
            Must be one of ['tip3p', 'tip4pew', 'tip5p', 'spce']
        constrained: bool
            Whether to use rigid water or not
        Examples
        --------
        Create water cluster with default settings
        >>> cluster = WaterCluster()
        >>> system, positions = cluster.system, cluster.positions
        """

        TestSystem.__init__(self, **kwargs)

        supported_models = ['tip3p', 'tip4pew', 'tip5p', 'spce']
        if model not in supported_models:
            raise Exception(
                "Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))

        # Load forcefield for solvent model and ions.
        ff = app.ForceField(model + '.xml')

        # Create empty topology and coordinates.
        top = app.Topology()
        pos = unit.Quantity((), unit.angstroms)

        # Create new Modeller instance.
        modeller = app.Modeller(top, pos)

        # Add solvent
        modeller.addSolvent(ff, model=model, numAdded=n_waters)

        # Get new topology and coordinates.
        new_top = modeller.getTopology()
        new_pos = modeller.getPositions()

        # Convert positions to numpy.
        positions = unit.Quantity(numpy.array(new_pos / new_pos.unit), new_pos.unit)

        # Create OpenMM System.
        system = ff.createSystem(new_top,
                                 nonbondedCutoff=openmm.NonbondedForce.NoCutoff,
                                 constraints=None,
                                 rigidWater=constrained,
                                 removeCMMotion=False)

        n_atoms = system.getNumParticles()
        self.ndof = 3 * n_atoms - 3 * constrained

        # Add a restraining potential centered at the origin.
        energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
        energy_expression += 'K = %f;' % (K / (unit.kilojoules_per_mole / unit.nanometers ** 2))  # in OpenMM units
        force = openmm.CustomExternalForce(energy_expression)
        for particle_index in range(n_atoms):
            force.addParticle(particle_index, [])
        system.addForce(force)

        self.topology = modeller.getTopology()
        self.system = system
        self.positions = positions


n_waters = 20

testsystem = WaterCluster(n_waters=n_waters, constrained=True)
(topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions

water_cluster_rigid = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=topology, system=system, positions=positions,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="water_cluster_rigid")

testsystem = WaterCluster(n_waters=n_waters, constrained=False)
(topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions

water_cluster_flexible = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=topology, system=system, positions=positions,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="water_cluster_flexible")


# bigger
n_waters = 40

testsystem = WaterCluster(n_waters=n_waters, constrained=True)
(topology, system, positions) = testsystem.topology, testsystem.system, testsystem.positions

big_water_cluster_rigid = EquilibriumSimulator(platform=configure_platform("Reference"),
                                           topology=topology, system=system, positions=positions,
                                           temperature=temperature,
                                           timestep=1.0 * unit.femtosecond,
                                           burn_in_length=100000, n_samples=1000,
                                           thinning_interval=10000, name="big_water_cluster_rigid")