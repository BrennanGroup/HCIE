import os
from hcie.molecule import Molecule
from hcie.mol2_dict import NO_SUBSTRUCTS, MOLECULE_TYPE
from tests.test_utils import work_in_tmp_dir
here = os.path.dirname(os.path.abspath(__file__))

CHARGE_TYPE = 'USER_CHARGES'


def test_string_representation():
    test_mol = Molecule('CC(C)=O', name='acetone')
    assert(str(test_mol)) == "Molecule(acetone, smiles=CC(C)=O)"


def test_xyz_instantiation():
    path_to_test_file = os.path.join(here, 'Data', 'acetone.xyz')
    test_mol = Molecule(path_to_test_file, name='acetone')

    assert test_mol.num_atoms == 10
    assert test_mol.num_bonds == 9
    assert test_mol.coordinates[3] == [3.10249, 0.53629, -0.83831]


def test_mol2_filename():

    assert Molecule('c1ccccc1', name='benzene').mol2_filename == 'benzene.mol2'

    # If no name is provided, should default to query
    assert Molecule('C').mol2_filename == 'query.mol2'


def test_num_atoms():

    assert Molecule('c1ccccc1').num_atoms == 12

    # Test also for sp3 carbons
    assert Molecule('CCC').num_atoms == 11


def test_num_bonds():

    assert Molecule('C2=Nc1ccccc1C2').num_bonds == 17

    assert Molecule('C1CCC1').num_bonds == 12


def test_no_substructs():

    assert Molecule('c1ccccc1').no_substructs == NO_SUBSTRUCTS

    assert Molecule('C1CCC1').no_substructs == NO_SUBSTRUCTS


def test_molecule_type():

    assert Molecule('c1ccccc1').molecule_type == MOLECULE_TYPE

    assert Molecule('C1CCC1').molecule_type == MOLECULE_TYPE


def test_charge_type():

    assert Molecule('c1ccccc1').charge_type == CHARGE_TYPE

    assert Molecule('C1CCC1').charge_type == CHARGE_TYPE


def test_generate_sybyl_code():

    # Test aromatic case
    test_heterocycle = Molecule('c2cnc1nccnc1n2')

    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(9)) == 'N.ar'
    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(1)) == 'C.ar'
    assert test_heterocycle.generate_sybyl_code(test_heterocycle.GetAtomWithIdx(13)) == 'H'

    # Test aliphatic case

    but_2_one = Molecule('CC(=O)CC')

    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(0)) == 'C.3'
    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(2)) == 'O.2'
    assert but_2_one.generate_sybyl_code(but_2_one.GetAtomWithIdx(12)) == 'H'


def test_get_atom_name():

    assert Molecule('C').get_atom_name('C', 0) == 'C0'


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

    assert Molecule('C').sort_elements(test_dic) == sorted_test


def test_elements():

    assert Molecule('c2cnc1nccnc1n2').elements() == ['C', 'N', 'H']


def test_elements_by_index():

    elements_by_index = {
        'C': [0, 1, 3, 5, 6, 8],
        'N': [2, 4, 7, 9],
        'H': [10, 11, 12, 13]
    }

    assert Molecule('c2cnc1nccnc1n2').elements_by_index() == elements_by_index


@work_in_tmp_dir()
def test_making_mol2_file():
    xyz_path = os.path.join(here, 'Data', 'acetone.xyz')
    test_mol = Molecule(xyz_path)

    # To avoid calculation, these charges have already been calculated
    test_mol.charges = [-0.14, 0.246, -0.14, -0.363, 0.051, 0.074, 0.074, 0.051, 0.074, 0.074]
    test_mol.write_mol2_file()

    mol2_lines = open('acetone.mol2', 'r').readlines()
    assert len(mol2_lines) == 29

    mol2_charges = [float(line.split()[8]) for line in mol2_lines[7:17]]
    assert mol2_charges == test_mol.charges

    # Test that the atomic coordinates have been correctly copied into the mol2 file
    assert float(mol2_lines[8].split()[2]) == 2.486
