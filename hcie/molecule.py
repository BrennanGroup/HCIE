"""
Contains functionality for Molecule class - inherits from RDKit Molecule class instantiated from a SMILES string
"""

from rdkit import Chem


class Molecule(Chem.Mol):
    """
    Molecule class. Contains information about the molecule necessary to construct mol2 files, optimise geometries,
    and perform VEHICLe searches
    """

    def __init__(self,
                 smiles: str,
                 name: str = 'query',
                 mol2: str = None):
        super().__init__(Chem.AddHs(Chem.MolFromSmiles(smiles)))
        self.smiles = smiles
        self.name = name
        self.mol2 = mol2

    def __str__(self):
        return f'Molecule({self.name}, smiles={self.smiles})'


