"""
This is a script for visualising the top returned bioisosteres from HCIE in their alignment of best similarity.
It is written to be run in PyMol, and can be run as follows from the PyMol command line:

run pymol_visualisation.py
"""

# First load in the HCIE alignments from the SDF
cmd.load('overlay.sdf', multiplex=1)
util.cbag('all')
cmd.set('grid_mode', 1)

# ShaEP outputs the aligned isosteres with the most similar last, so this needs to be reversed
isosteres = cmd.get_object_list()
isosteres = list(reversed(isosteres))

for idx, isostere in enumerate(isosteres):
    cmd.set('grid_slot', idx + 1, isostere)

# Now set the view parameters
cmd.show('sticks')
cmd.set('ray_opaque_background', 'off')
cmd.set('stick_radius', 0.1)
cmd.show('spheres')
cmd.set('sphere_scale', 0.15, 'all')
cmd.set('sphere_scale', 0.12, 'elem H')
cmd.color('gray40', 'elem C')
cmd.set('sphere_quality', 30)
cmd.set('stick_quality', 30)
cmd.set('sphere_transparency', 0.0)
cmd.set('stick_transparency', 0.0)
cmd.set('ray_shadow', 'off')
cmd.set('orthoscopic', 1)
cmd.set('antialias', 2)
cmd.bg_color('white')
