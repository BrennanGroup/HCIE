"""
Dictionary of parameters for mol2 file generation
"""

# Dictionaries that map RDKit properties to those expected in TRIPOS Mol2 Files
bond_types = {
    'AROMATIC': 'ar',
    'SINGLE': '1',
    'DOUBLE': '2',
    'TRIPLE': '3',
}

# The SP3D entry is a patch for a bug in the xyz implementation of S169,
# which does not register as aromatic.
atom_types = {
    'SP': '1',
    'SP2': '2',
    'SP3': '3',
    'AROMATIC': 'ar',
    'S': '',
    'SP3D': '2'
}

# Some user-defined variables - at the moment these are hard-coded,
# I may find a better way of doing it later
NO_SUBSTRUCTS = 1
MOLECULE_TYPE = 'SMALL'
