"""
Tools for rendering molecules used in test systems.

"""

from time import sleep

testsystems_to_render = {
    ('AlanineDipeptideVacuum', 'sticks', 'all'),
    ('WaterBox', 'spheres', 'all'),
    ('LysozymeImplicit', 'cartoon', 'not hydrogen'),
    ('DHFRExplicit', 'cartoon', 'not (hydrogen or resn HOH)'),
    ('SrcExplicit', 'cartoon', 'not (hydrogen or resn HOH)'),
    }

def write_pdb_files():
    from openmmtools import testsystems
    from simtk.openmm import app
    for (name, representation, selection) in testsystems_to_render:
        # Create the test system
        testsystem = getattr(testsystems, name)()

        # Write the PDB file
        pdb_filename = name + '.pdb'
        with open(pdb_filename, 'w') as outfile:
            app.PDBFile.writeFile(testsystem.topology, testsystem.positions, outfile)

from pymol import cmd
def render_figure(name, representation, selection, width='4in', height='4in', dpi=300, zoom=+5):
    pdb_filename = name + '.pdb'
    png_filename = name + '.png'
    cmd.load(pdb_filename)
    cmd.hide('all')
    #cmd.color('green', 'all')
    #cmd.util.chainbow(selection)
    #cmd.util.cbag('all')
    cmd.show(representation, selection)
    cmd.orient('all')
    cmd.dss('all')
    cmd.zoom(selection, zoom)
    cmd.bg_color('white')
    cmd.set('ray_opaque_background', 0)
    cmd.png(png_filename, width, height, dpi, ray=1)

cmd.extend("render_figure",render_figure)

#if __name__ == '__main__':
    #write_pdb_files()
