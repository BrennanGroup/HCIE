"""
Contains functionality for Molecule class - inherits from RDKit Molecule class instantiated from a SMILES string
"""
from hcie.mol2_dict import bond_types, atom_types, NO_SUBSTRUCTS, MOLECULE_TYPE
from rdkit import Chem
import csv
import autode
import subprocess
import pkg_resources

# Set path to package data files referenced within the code
VEHICLE_MOL2 = pkg_resources.resource_filename('hcie', 'Data/vehicle_dft.mol2')


class Molecule(Chem.Mol):
    """
    Molecule class. Contains information about the molecule necessary to construct mol2 files, optimise geometries,
    and perform VEHICLe searches
    """

    def __init__(self,
                 smiles: str,
                 name: str = 'query',
                 mol2: str = None,
                 output_queries: bool = True,
                 max_hits: int = 20):
        super().__init__(Chem.AddHs(Chem.MolFromSmiles(smiles)))
        self.smiles = smiles
        self.name = name
        self.mol2 = mol2
        self.output_queries = output_queries
        self.max_hits = max_hits

        # These attributes will be assigned after the geometry optimisation
        self.coordinates: list = []
        self.charges: list = []

        # Necessary attributes for generating TRIPOS mol2 files needed by SHaEP
        # The TRIPOS mol2 index is not the same as the RDKit index, so a mapping between the two is necessary
        self._atoms_by_tripos_order: list = self.set_atoms_by_tripos_order()
        self._rdkit_to_tripos_lookup: dict = self.set_rdkit_to_tripos_lookup()

        self.bond_types: dict = bond_types
        self.atom_types: dict = atom_types

    def __str__(self):
        return f'Molecule({self.name}, smiles={self.smiles})'

    def set_coordinates(self, mol: autode.Molecule):
        """
        Sets the coordinates attribute using optimised coordinates from the autode.Molecule instance
        :param mol: instance of autode.Molecule class, after geometry optimisation
        :return: list of lists of atomic coordinates, ordered by atom index
        """
        coords = mol.coordinates.tolist()

        self.coordinates = coords

        return coords

    @staticmethod
    def test_coordinate_row(atom_symbol, coordinate_row):
        """
        Tests that the atom indexing has been preserved across the geometry optimisation. Compares the atom symbol from
        the RDKit molecule to that in the xyz file and checks that they are the same
        :param atom_symbol: atomic symbol extracted from RDKIt molecule instance
        :param coordinate_row: list. coordinate row from xyz file
        :return: True if same, False if not
        """
        if atom_symbol == coordinate_row[0]:
            return True
        else:
            return False

    @staticmethod
    def open_file(filename):
        """
        opens file and returns a list of contents
        :param filename: path of file to open
        :return: list of file contents
        """
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            return list(csv_reader)

    def optimise(self):
        """
        Optimises the geometry and calculates the partial charges. Prints xyz file of optimised atomic co-ordinates
        :return:
        """
        mol = autode.Molecule(name=self.name, smiles=self.smiles)
        mol.optimise(method=autode.methods.XTB())

        self.charges = self.calculate_charges(mol)

        self.set_coordinates(mol)

        return None

    def calculate_charges(self, molecule):
        """
        Calculate the atomic partial charges in the molecule using Mulliken population analysis
        :param molecule: instance of autode.Molecule class to calculate partial charges
        :return: list of atomic partial charges, ordered by atomic index
        """
        calc = autode.Calculation(name=f'{self.name}_calc', method=autode.methods.XTB(),
                                  keywords=autode.Config.XTB.keywords.opt, molecule=molecule)
        calc.run()
        atomic_charges = calc.get_atomic_charges()

        return atomic_charges

    @property
    def mol2_file(self) -> str:
        """
        Path to mol2 file for instance of molecule
        :return: string
        """
        return f'{self.name}.mol2'

    @property
    def num_atoms(self):
        return len(self.GetAtoms())

    @property
    def num_bonds(self):
        return len(self.GetBonds())

    def write_mol2_molecule_block_to_file(self):
        """
        Calculates and returns, in the correct format, the '@<TRIPOS>MOLECULE'
        block for the instance of the molecule.
        The number of features and sets are hard-coded to 0
        """
        with open(self.mol2_file, 'a') as mol2_file:
            print(f'@<TRIPOS>MOLECULE',
                  f'{self.name}',
                  f'{self.num_atoms} {self.num_bonds} {self.no_substructs} 0 0 ',
                  f'{self.molecule_type}',
                  f'{self.charge_type}',
                  '', sep='\n', file=mol2_file)

        return None

    @property
    def no_substructs(self):
        """
        Number of substructures in the molecule, for now hardcoded to 1
        :return: int 1
        """
        return NO_SUBSTRUCTS

    @property
    def molecule_type(self):
        """
        Molecule type, this is defined in mol2_dict.py
        :return: str - molecule type
        """
        return MOLECULE_TYPE

    @property
    def charge_type(self):
        """
        Type of charge used for mol2 file. Hardcoded to USERCHARGES
        :return: string
        """
        return 'USER_CHARGES'

    def generate_sybyl_code(self, atom) -> str:
        """
        Generate the sybyl code for each atom - comprised of the symbol and
        atom type e.g. aromatic/sp3 etc.
        The atom_types dictionary is defined in mol2_dict.py
        :param atom: instance of RDKit atom object in RDKit mol object
        :return: string of sybyl code for atom
        """
        symbol = atom.GetSymbol()

        if symbol == 'C' or symbol == 'N':
            if atom.GetIsAromatic():
                sybyl = f'{symbol}.ar'
            else:
                sybyl = f'{symbol}.{self.atom_types[str(atom.GetHybridization())]}'
        elif symbol == 'H':
            sybyl = 'H'
        else:
            sybyl = f'{symbol}.{self.atom_types[str(atom.GetHybridization())]}'

        return sybyl

    @staticmethod
    def get_atom_name(symbol, index_label) -> str:
        """
        Returns the TRIPOS atom name
        :param symbol: atom symbol
        :param index_label: TRIPOS atom index label
        :return: name of atom
        """
        return f'{symbol}{index_label}'

    @staticmethod
    def sort_elements(element_dict) -> list:
        """
        Sorts a dictionary of elements into TRIPOS format - increasing
        atomic number with H at the end
        :param element_dict: dictionary keys = elements,
        values = atomic numbers - values can be discarded
        :return: list sorted into Tripos order
        """
        element_dict = dict((sorted(element_dict.items(), key=lambda item: item[1])))
        elements = [key for key in element_dict]

        # H needs to be at the end of the list of elements
        if 'H' in elements:
            elements.remove('H')
            elements.append('H')

        return elements

    def elements(self) -> list:
        """
        Returns a list of all the elements present in the input molecule,
        in increasing atomic number order with H last
        :return: list of element symbols in order of increasing atomic number, with H at the end.
        """
        element_dict = {}
        for atom in self.GetAtoms():
            if atom.GetSymbol() not in element_dict:
                element_dict[atom.GetSymbol()] = atom.GetAtomicNum()

        elements = self.sort_elements(element_dict)

        return elements

    def elements_by_index(self) -> dict:
        """
        Returns a dictionary with element symbols as keys, and the atom ID's
        of those elements in the molecule as values
        """
        mol_by_elements = {}
        for atom in self.GetAtoms():
            symbol = atom.GetSymbol()

            if symbol not in mol_by_elements:
                mol_by_elements[symbol] = []

            mol_by_elements[symbol].append(atom.GetIdx())

        return mol_by_elements

    def set_atoms_by_tripos_order(self) -> list:
        """
        Generates a list of RDKit atom indices in the correct Tripos format. I.e. ordered by element atomic number, with
        H at the end.
        :return: list of RDKit atom indicies in TRIPOS index order
        """
        tripos_sorted_dict = {}
        elements = self.elements()
        mol_by_elements = self.elements_by_index()

        for element in elements:
            tripos_sorted_dict[element] = mol_by_elements[element]

        return sum(tripos_sorted_dict.values(), [])

    def write_mol2_atom_block_to_file(self) -> None:
        """
        Computes the <TRIPOS>ATOM block for the molecule instance. For each atom in the molecule, a row will contain
        all necessary information:

                        atom_id atom_name x y z atom_type substructure_id substructure_name charge

        atom_id (integer) = the ID number of the atom at the time the file was created.
        atom_name (string) = the name of the atom.
        x (real) = the x coordinate of the atom.
        y (real) = the y coordinate of the atom.
        z (real) = the z coordinate of the atom.
        atom_type (string) = the SYBYL atom type for the atom.
        subst_id (integer) = the ID number of the substructure containing the atom.
        subst_name (string) = the name of the substructure containing the atom.
        charge (real) = the charge associated with the atom.
        :return: None
        """
        with open(self.mol2_file, 'a') as mol2_file:
            print('@<TRIPOS>ATOM', file=mol2_file)
            for idx, atom_idx in enumerate(self._atoms_by_tripos_order, 1):
                atom = self.GetAtomWithIdx(atom_idx)
                row = self.get_atom_row(atom, idx)
                print(row, file=mol2_file)

        return None

    def get_atom_row(self, atom, idx) -> str:
        """
        Prints the atom row for atom, including SYBYL index idx
        :param atom: RDKit atom object
        :param idx: SYBYL index of atom
        :return: String
        """
        atom_name = self.get_atom_name(atom.GetSymbol(), idx)
        x, y, z = self.coordinates[atom.GetIdx()]
        sybyl = self.generate_sybyl_code(atom)
        substruct_no = 1
        substruct_name = '****'
        charge = self.charges[atom.GetIdx()]

        row = f'{idx:<8}{atom_name:<11}{x:>7.3f}{y:>10.3f}{z:>10.3f} {sybyl:<10} ' \
              f'{substruct_no} {substruct_name:<11}{charge:>7.4f} '

        return row

    def set_rdkit_to_tripos_lookup(self) -> dict:
        """
        Generates a dictionary that maps the RDKit indices of all the atoms in the molecule to their TRIPOS ID's for the
        purposes of generating Mol2 files.
        :return: Dictionary - keys are RDKit indices, values are TRIPOS indices
        """
        lookup = {rdkit_idx: tripos_idx for tripos_idx, rdkit_idx in enumerate(self._atoms_by_tripos_order, 1)}

        return lookup

    def write_mol2_bond_block_to_file(self) -> None:
        """
        Writes the <TRIPOS> bond block for the instance of the Molecule class to the mol2 file in the format:

                                bond_id origin_atom_id target_atom_id bond_type

        bond_id (integer) = the ID number of the bond at the time the file was
                            created. This is provided for reference only and is not used when the
                            .mol2 file is read into SYBYL.
        origin_atom_id (integer) = the ID number of the atom at one end of the bond.
        target_atom_id (integer) = the ID number of the atom at the other end of the bond.
        bond_type (string) = the SYBYL bond type (see below).

        Bond Types:
        • 1 = single
        • 2 = double
        • 3 = triple
        • am = amide
        • ar = aromatic
        • du = dummy
        • un = unknown (cannot be determined from the parameter tables)
        • nc = not connected

        for each bond
        :return: None
        """
        with open(self.mol2_file, 'a') as mol2_file:
            print('@<TRIPOS>BOND', file=mol2_file)
            for tripos_bond_idx, bond in enumerate(self.GetBonds(), 1):
                origin_atom_id = self._rdkit_to_tripos_lookup[bond.GetBeginAtom().GetIdx()]
                target_atom_id = self._rdkit_to_tripos_lookup[bond.GetEndAtom().GetIdx()]
                bond_type = self.bond_types[str(bond.GetBondType())]

                print(f'{tripos_bond_idx:<6}{origin_atom_id:<5}{target_atom_id:<5} {bond_type}', file=mol2_file)

        return None

    def write_mol2_substructure_block_to_file(self) -> None:
        """
        writes the <TRIPOS> substructure block for the instance of the molecule class to the mol2 file. As this package
        is designed for small molecules, this is hardcoded. The below is included for reference only.

            Each data record associated with this RTI consists of a single data line. The data
            line contains the substructure ID, name, root atom of the substructure,
            substructure type, dictionary type, chain type, subtype, number of inter
            substructure bonds, SYBYL status bits, and user defined comment.
        :return: None
        """
        with open(self.mol2_file, 'a') as mol2_file:
            print('@<TRIPOS>SUBSTRUCTURE', file=mol2_file)
            print('     1 ****        1 GROUP             0       ****    0 ROOT', file=mol2_file)

        return None

    def write_mol2_file(self) -> None:
        """
        Writes instance of molecule to mol2 file
        :return: None
        """
        self.write_mol2_molecule_block_to_file()
        self.write_mol2_atom_block_to_file()
        self.write_mol2_bond_block_to_file()
        self.write_mol2_substructure_block_to_file()

        return None

    def search_shaep(self) -> None:
        """
        Runs a ShaEP search of the query molecule in the VEHICLe database
        :return: None
        """
        if self.output_queries is True:
            subprocess.run(['shaep', '--maxhits', str(self.max_hits), '-q', self.mol2_file, '--output-file',
                            'similarity.txt',
                            '--structures',
                            'overlay.sdf', '--outputQuery', VEHICLE_MOL2])
        else:
            subprocess.run(['shaep', '--maxhits', str(self.max_hits), '-q', self.mol2_file, '--output-file',
                            'similarity.txt',
                            '--structures',
                            'overlay.sdf', VEHICLE_MOL2])

        return None


if __name__ == '__main__':
    mol = Molecule(smiles='c2cc1nccnc1cn2', name='test_het')
    mol.optimise()
    mol.write_mol2_file()
    mol.search_shaep()
