import pytest
from rdkit import Chem
from hcie.molecule import Molecule
from hcie.mol2_dict import NO_SUBSTRUCTS, MOLECULE_TYPE, atom_types, bond_types

CHARGE_TYPE = 'USER_CHARGES'


def test_mol2_filename():

    assert Molecule(smiles='c1ccccc1', name='benzene').mol2_filename == 'benzene.mol2'

    # If no name is provided, should default to query
    assert Molecule(smiles='C').mol2_filename == 'query.mol2'


def test_num_atoms():

    assert Molecule(smiles='c1ccccc1').num_atoms == 12

    # Test also for sp3 carbons
    assert Molecule(smiles='CCC').num_atoms == 11


def test_num_bonds():

    assert Molecule(smiles='C2=Nc1ccccc1C2').num_bonds == 17

    assert Molecule(smiles='C1CCC1').num_bonds == 12


def test_no_substructs():

    assert Molecule(smiles='c1ccccc1').no_substructs == NO_SUBSTRUCTS

    assert Molecule(smiles='C1CCC1').no_substructs == NO_SUBSTRUCTS


def test_molecule_type():

    assert Molecule(smiles='c1ccccc1').molecule_type == MOLECULE_TYPE

    assert Molecule(smiles='C1CCC1').molecule_type == MOLECULE_TYPE


def test_charge_type():

    assert Molecule(smiles='c1ccccc1').charge_type == CHARGE_TYPE

    assert Molecule(smiles='C1CCC1').charge_type == CHARGE_TYPE


def test_generate_sybyl_code():

    # Test aromatic case
    test_heterocycle = Molecule(smiles='c2cnc1nccnc1n2')

    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(9)) == 'N.ar'
    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(1)) == 'C.ar'
    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(13)) == 'H'

    # Test aliphatic case

    but_2_one = Molecule(smiles='CC(=O)CC')

    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(0)) == 'C.3'
    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(2)) == 'O.2'
    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(12)) == 'H'


def test_get_atom_name():

    assert Molecule(smiles='C').get_atom_name('C', 0) == 'C0'


def test_sort_elements():

    test_dic = {
        'O': 8,
        'C': 6,
        'Pb': 82,
        'H': 1,
        'Ge': 32,
        'Cs': 55
    }

    sorted_test = ['C', 'O', 'Ge', 'Cs', 'Pb', 'H']

    assert Molecule(smiles='C').sort_elements(test_dic) == sorted_test


def test_elements():

    assert Molecule(smiles='c2cnc1nccnc1n2').elements() == ['C', 'N', 'H']


def test_elements_by_index():

    elements_by_index = {
        'C': [0, 1, 3, 5, 6, 8],
        'N': [2, 4, 7, 9],
        'H': [10, 11, 12, 13]
    }

    assert Molecule(smiles='c2cnc1nccnc1n2').elements_by_index() == elements_by_index
