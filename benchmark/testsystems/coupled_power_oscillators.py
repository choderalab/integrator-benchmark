import numpy as np

from simtk import openmm
from simtk import unit
from simtk.openmm import app

from openmmtools.testsystems import TestSystem

from benchmark.testsystems.bookkeepers import EquilibriumSimulator
from benchmark.testsystems.configuration import configure_platform


class CoupledPowerOscillators(TestSystem):
    """Create a 3D grid of power oscillators.
     Each particle is in an isotropic (x^b + y^b + z^b) well.
     Each neighboring pair of particles is connected by a harmonic bond force.
    """

    def __init__(self, nx=3, ny=3, nz=3,
                 K=100.0, b=2.0, mass=39.948 * unit.amu, well_radius=1.0,
                 grid_spacing=1.0 * unit.angstrom,
                 coupling_strength=100.0 * unit.kilocalories_per_mole / unit.angstrom,
                 **kwargs):
        """Initialize particles on a 3D grid of specified size.

        Parameters
        ----------
        nx, ny, nz : ints
            number of particles in x, y, z directions, respectively
        K : float
            well depth
        b : float
            exponent
        mass : simtk.unit.Quantity
            particle mass
        grid_spacing : simtk.unit.Quantity
            increment between grid points
        coupling_strength : simtk.unit.quantity
            strength of HarmonicBond force between each neighboring pair of particles

        Attributes
        ----------
        topology, positions, system
        """

        # 1. Initialize
        # 2. Set particles on a 3D grid, use a CustomExternalForce to place each particle in a well
        # 3. Add a HarmonicBondForce to each neighboring pair of particles.


        ### 1. Initialize

        TestSystem.__init__(self, **kwargs)
        # Set unit of well-depth appropriately
        #K *= unit.kilocalories_per_mole / unit.angstroms ** b

        # Determine total number of atoms.
        natoms = nx * ny * nz

        # Create an empty system object.
        system = openmm.System()

        ### 2. Put particles on a 3D grid

        positions = unit.Quantity(np.zeros([natoms, 3], np.float32), unit.angstrom)

        atom_index = 0

        # Store the atom indices in a 3-way array
        # so that we can conveniently determine nearest neighbors
        atom_indices = np.zeros((nx, ny, nz), dtype=int)
        energy_expression = "{K} * (sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / {r})^{b};".format(
            b=b, K=K, r=well_radius)
        force = openmm.CustomExternalForce(energy_expression)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    system.addParticle(mass)

                    xyz = (grid_spacing.value_in_unit(unit.angstrom)) * np.array([i, j, k], dtype=float)
                    positions[atom_index] = (xyz + np.random.randn(3)*0.01) * unit.angstrom

                    atom_indices[i, j, k] = atom_index

                    # Add this particle's well
                    force.addParticle(atom_index, xyz)

                    atom_index += 1
        system.addForce(force)

        ### 3. Couple each pair of neighbors

        # Find each pair of neighbors in this grid.
        # Connect each particle (i,j,k), to its "forward" neighbors,
        #   (i+1, j, k), (i, j+1, k), (i, j, k+1),
        # (if they exist)
        bonds = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i + 1) < nx:
                        bonds.append((atom_indices[i, j, k], atom_indices[i + 1, j, k]))
                    if (j + 1) < ny:
                        bonds.append((atom_indices[i, j, k], atom_indices[i, j + 1, k]))
                    if (k + 1) < nz:
                        bonds.append((atom_indices[i, j, k], atom_indices[i, j, k + 1]))

        # Add these HarmonicBondForces to the system
        force = openmm.HarmonicBondForce()
        for bond in bonds:
            force.addBond(int(bond[0]), int(bond[1]), grid_spacing, coupling_strength)
        system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)

        # Set topology, positions, system as instance attributes
        self.topology = topology
        self.positions = positions
        self.system = system

temperature = 298 * unit.kelvin
testsystem = CoupledPowerOscillators(nx=5, ny=5, nz=5)
top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
coupled_power_oscillators = EquilibriumSimulator(platform=configure_platform("CPU"),
                                           topology=top, system=sys, positions=pos,
                                           temperature=temperature,
                                           ghmc_timestep=1.0 * unit.femtosecond,
                                           burn_in_length=1000, n_samples=1000,
                                           thinning_interval=10, name="coupled_power_oscillators")
