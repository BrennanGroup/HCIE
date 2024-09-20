import numpy as np
import pytest

from multi_vector.molecule import Molecule


def test_smiles_handling():
    # Test with a valid SMILES string
    pyridine = Molecule('c1ncccc1')
    assert pyridine.smiles == 'c1ncccc1'
    assert pyridine.mol is not None

    # Test with an invalid SMILES string
    with pytest.raises(ValueError):
        mol = Molecule('c1cCoc1')


def test_molecule_initialization():
    pyridine = Molecule('c1ncccc1')

    # Test that H atoms are correctly added when query is instantiated
    atoms = [atom.GetSymbol() for atom in pyridine.mol.GetAtoms()]
    assert len(atoms) == 11
    assert pyridine.mol.GetAtomWithIdx(1).GetSymbol() == 'N'

    conf = pyridine.mol.GetConformer()
    assert conf is not None
    assert conf.GetNumAtoms() == pyridine.mol.GetNumAtoms()


def test_coordinate_extraction():
    furan = Molecule('c1occc1')
    coords = furan.coords
    assert coords.shape == (9, 3)


def test_coordinate_validity():
    furan = Molecule('c1occc1')
    coords = furan.coords
    assert np.all(np.isfinite(coords))


def test_exit_vector_identification():
    furan = Molecule('c1occc1')
    exit_vectors = furan.exit_vectors
    assert len(exit_vectors) == 4
    for exit_vector in exit_vectors:
        assert len(exit_vector) == 2  # Each exit vector should be comprised of 2 atoms, a base atom and a tail atom

        assert furan.mol.GetAtomWithIdx(exit_vector[0]).GetSymbol() != 'H'
        assert furan.mol.GetAtomWithIdx(exit_vector[1]).GetSymbol() == 'H'


def test_no_exit_vectors():
    weird_het = Molecule('n1nnnnn1')
    assert len(weird_het.exit_vectors) == 0


def test_exit_vector_pairs():
    quinoline = Molecule('c2ccc1ncccc1c2')
    assert len(quinoline.exit_vectors) == 7
    assert len(quinoline.exit_vector_pairs) == 21

    for vector_pair in quinoline.exit_vector_pairs:
        assert len(vector_pair) == 2
        for pair in vector_pair:
            assert len(pair) == 2


def test_distance_between_atoms():
    furan = Molecule('c1occc1')
    distance = furan.get_distance_between_atoms(2, 3)  # Distance between adjacent aromatic carbons
    assert float(np.round(distance, decimals=1)) == 1.4  # the distance between these atoms should be around 1.4 Å
    for bond in furan.mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if 'H' in (begin_atom.GetSymbol(), end_atom.GetSymbol()):
            x_h_bond_distance = furan.get_distance_between_atoms(begin_atom.GetIdx(), end_atom.GetIdx())
            assert np.round(x_h_bond_distance, decimals=1) == 1.1  # All C-H bonds should be at 1.1 Å


def test_same_atom_distance():
    furan = Molecule('c1occc1')
    assert furan.get_distance_between_atoms(2, 2) == pytest.approx(0)


def test_angle_between_3_atoms():
    benzene = Molecule('c1ccccc1')
    assert benzene.get_angle_between_three_atoms(6, 0, 1) == pytest.approx(120, abs=0.1)
    assert benzene.get_angle_between_three_atoms(0, 1, 2) == pytest.approx(120, abs=0.1)


def test_exit_vector_measures():
    furan = Molecule('c1occc1')
    assert isinstance(furan.exit_vector_properties, list)
    assert len(furan.exit_vector_properties) == len(furan.exit_vector_pairs)
    for exit_vector_dict in furan.exit_vector_properties:
        assert 'distance' in exit_vector_dict
        assert 'angles' in exit_vector_dict
        assert 'av' in exit_vector_dict['angles']
        assert 'a1' in exit_vector_dict['angles']
        assert 'a2' in exit_vector_dict['angles']

        assert exit_vector_dict['angles']['a1'] < exit_vector_dict['angles']['a2']


def test_angle_hash():
    furan = Molecule('c1occc1')  # The exact nature of this query does not matter, but one is needed to get hashes
    test_angle_hashes = {5: '000',
                         20: '001',
                         80: '010',
                         85: '011',
                         136: '100',
                         170: '101',
                         180: '101'}

    for angle, expected_hash in test_angle_hashes.items():
        assert furan.get_angle_hash(angle) == expected_hash


def test_distance_hash():
    furan = Molecule('c1occc1')  # The exact nature of this query does not matter, but one is needed to get hashes
    test_distance_hashes = {1.5: '00000',
                            2.0: '00001',
                            2.2: '00001',
                            2.4: '00010',
                            2.6: '00011',
                            2.8: '00100',
                            3.1: '00101',
                            3.3: '00110',
                            3.5: '00111',
                            3.8: '01000',
                            4.1: '01001',
                            4.3: '01010',
                            4.5: '01011',
                            4.8: '01100',
                            5.1: '01101',
                            5.3: '01110',
                            5.5: '01111',
                            5.8: '10000',
                            6.0: '10000',
                            14.0: '10001'
                            }
    for distance, expected_hash in test_distance_hashes.items():
        assert furan.get_distance_hash(distance) == expected_hash
