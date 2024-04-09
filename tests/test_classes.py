import unittest
from hcie.classes import Molecule


class TestMolecule(unittest.TestCase):
    def setUp(self):
        self.smiles = 'c1ccncc1'
        self.name = 'pyridine'
        self.molecule = Molecule(self.smiles, name=self.name)

    def test_initialization(self):
        self.assertEqual(self.molecule.smiles, self.smiles)
        self.assertEqual(self.molecule.name, self.name)
        self.assertIsInstance(self.molecule.functionalisable_bonds, list)
        self.assertIsInstance(self.molecule.shape_scores, dict)
        self.assertIsInstance(self.molecule.esp_scores, dict)
        self.assertIsInstance(self.molecule.total_scores, list)
        self.assertEqual(len(self.molecule.functionalisable_bonds), 5)
        self.assertEqual(len(self.molecule.shape_scores), 0)
        self.assertEqual(len(self.molecule.esp_scores), 0)
        self.assertEqual(len(self.molecule.total_scores), 0)
        self.assertIsNotNone(self.molecule.rdmol)
        self.assertIsNotNone(self.molecule.functionalisable_bonds)
        self.assertIsNotNone(self.molecule.charges)
        self.assertEqual(self.molecule.num_conformers, 10)
        self.assertEqual(self.molecule.num_conformers, len(self.molecule.coords))
        self.assertEqual(len(self.molecule.coords), 10)

    def test_coords(self):
        self.assertEqual(len(self.molecule.coords[0]), self.molecule.rdmol.GetNumAtoms())

    def test_load_rdkit_mol_from_smiles(self):
        self.assertEqual(self.molecule.rdmol.GetNumAtoms(), 11)

    def test_mol_block(self):
        self.assertEqual(len(self.molecule.mol_block), 10)

    def test_get_functionalisable_bonds(self):
        self.assertIn((1, 7), self.molecule.functionalisable_bonds)
        self.assertNotIn((1, 2), self.molecule.functionalisable_bonds)
