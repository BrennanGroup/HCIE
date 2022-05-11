"""
Generates mol2 file for the query molecule from information and files generated in optimise.py
"""
import os

from hcie.mol2_dict import bond_types, atom_types, NO_SUBSTRUCTS, MOLECULE_TYPE
from hcie.molecule import Molecule

from csv import reader
import numpy as np
import pandas as pd


class Mol2(Molecule):
    """
    Class used to represent a molecule in mol2 format, including the atom, molecule, and bond blocks, and the
    methodology to generate and save .mol2 files for an instance of the class
    """

    def __init__(self,
                 smiles: str,
                 name: str = 'query'):
        """
        Mol2 class.
        :param smiles: SMILES string of instance of molecule
        :param name: Name of molecule, defaults to 'query'
        """
        super().__init__(smiles=smiles, name=name)
        self.index_lookup = None
        self.charges = self.get_external_charges()
        self.coordinates = self.get_xyz_coordinates()

    def molecule_block(self):
        """
        Calculates and returns, in the correct format, the '@<TRIPOS>MOLECULE'
        block for the instance of the molecule.
        The number of features and sets are hard-coded to 0
        """
        charge_type = 'USER_CHARGES'
        name = self.name

        molecule_block = (f'@<TRIPOS>MOLECULE\n'
                          f'{name}\n'
                          f'{self.num_atoms} {self.num_bonds} {NO_SUBSTRUCTS} 0 0\n '
                          f'{MOLECULE_TYPE}\n'
                          f'{charge_type}\n')

        return molecule_block

    def atom_block(self):
        """
        Computes and returns, in the correct format, the '@<TRIPOS>ATOM
        block for the instance of the molecule.
        """

        tripos_atom = pd.DataFrame(
            columns=['rdkit_index', 'atom_name', 'x_coord', 'y_coord',
                     'z_coord', 'sybyl', 'substruct', 'substruct_name',
                     'partial_charge', 'atom_symbol', 'atom_index_label'])

        substruct_name = '****'

        for atom in self.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            charge = self.charges[atom_idx]

            # Get co-ordinates for each atom
            atom_coords = self.coordinates[atom_idx]
            x, y, z = self.get_coordinates_from_list(atom_coords)

            atom_index_label = self.get_atom_index_label(symbol, atom_idx)
            atom_name = self.get_atom_name(symbol, atom_index_label)

            sybyl = self.generate_sybyl_code(atom)

            # Append atom to dataframe of atomic information
            tripos_atom = tripos_atom.append(
                {'rdkit_index': atom_idx, 'atom_name': atom_name,
                 'x_coord': "%.4f" % x,
                 'y_coord': "%.4f" % y, 'z_coord': "%.4f" % z,
                 'sybyl': sybyl, 'substruct': NO_SUBSTRUCTS,
                 'substruct_name': substruct_name,
                 'partial_charge': "%.4f" % charge,
                 'atom_symbol': symbol, 'atom_index_label': atom_index_label},
                ignore_index=True)

        tripos_atom = self.sort_atom_dataframe(tripos_atom)

        # TRIPOS atom ID and RDKit index are not the same, need to generate
        # a mapping from one to t'other
        self.index_lookup = self.generate_index_mapper(tripos_atom)

        # Generate final dataframe, and return as a string
        atom_block = '@<TRIPOS>ATOM\n'
        for idx in tripos_atom.index:
            row = self.print_atom_row(idx, tripos_atom)
            atom_block += row + '\n'

        return atom_block

    @staticmethod
    def print_atom_row(idx, dataframe):
        """
        Takes a row of atom information in dataframe and generates a correctly formatted string of atom info
        :param idx: index of atom
        :param dataframe: dataframe containing atom information
        :return: string, row of atom block
        """
        index_chunk = idx
        name_chunk = dataframe["atom_name"][idx]
        x_chunk = dataframe["x_coord"][idx]
        y_chunk = dataframe["y_coord"][idx]
        z_chunk = dataframe["z_coord"][idx]
        sybyl_chunk = dataframe["sybyl"][idx]
        substruct_no = 1
        substruct_name = dataframe["substruct_name"][idx]
        charge_chunk = dataframe["partial_charge"][idx]

        row = f'{index_chunk:>7} {name_chunk:<11}{x_chunk:>7}{y_chunk:>10}{z_chunk:>10} {sybyl_chunk:<10} ' \
              f'{substruct_no} {substruct_name:<11}{charge_chunk:>7} '

        return row

    def bond_block(self):
        """
        Generates and returns the TRIPOS Bond Block for the instance of the
        molecule - ATOM Block must have been generated first for this to work
        :return: string - TRIPOS bond block for molecule
        """

        # atom block must have been created for bond block to be created
        if self.index_lookup is None:
            print('Atom Block must be generated before Bond Block')
            return

        bond_block = self.get_bond_dataframe()

        # Convert RDKit indices to TRIPOS atom ID numbers so there is
        # consistency between ATOM block and BOND block
        sorted_block = self.sort_bond_dataframe(bond_block)

        bond_block = '@<TRIPOS>BOND\n'

        for idx in sorted_block.index:
            row = self.print_bond_row(idx, sorted_block)
            bond_block += row + '\n'

        return bond_block

    @staticmethod
    def substructure_block():
        """
        Returns TRIPOS Substructure block - this is hardcoded in for now
        :return: string
        """
        substructure_block = '@<TRIPOS>SUBSTRUCTURE\n'
        substructure_block += '     1 ****        1 GROUP             0       ****    0 ROOT'

        return substructure_block

    @staticmethod
    def print_bond_row(idx, dataframe):
        """
        Prints the correctly formatted row of the @TRIPOS BOND block for each bond
        :param idx: index of bond
        :param dataframe: dataframe of bond information
        :return: string - row of TRIPOS Bond block
        """
        bond_id = idx
        begin_atom = dataframe["begin"][idx]
        end_atom = dataframe["end"][idx]
        bond_type = dataframe["bond_type"][idx]

        bond_row = f'{bond_id:>6}{begin_atom:>5}{end_atom:>5} {bond_type}'

        return bond_row

    def print_mol2_file(self):
        """
        Saves the current instance of the molecule to a mol2 file in the
        current directory
        """
        filename = f'{self.name}.mol2'

        block = self.molecule_block() + self.atom_block() + self.bond_block() + self.substructure_block()

        with open(filename, 'w') as file:
            file.write(block)

        return None

    @staticmethod
    def open_file(filename):
        """
        opens file and returns a list of contents
        :param filename: path of file to open
        :return: list of file contents
        """
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            return list(csv_reader)

    @property
    def charge_file(self):
        """
        Returns path to charge csv file
        :return:
        """
        return os.path.join('..', 'autodE_outputs', f'{self.name}_charges.csv')

    def get_external_charges(self):
        """
        Returns a list of partial charges from a csv file of partial charges calculated with autodE
        :return: list of atomic partial charges, sorted by RDKit atom index as defined by SMILES string
        """
        all_charges = self.open_file(self.charge_file)

        charges = [float(charge) for charge in all_charges[0]]

        return charges

    @property
    def num_atoms(self):
        return len(self.GetAtoms())

    @property
    def num_bonds(self):
        return len(self.GetBonds())

    @staticmethod
    def sort_elements(element_dict):
        """
        Sorts a dictionary of elements into TRIPOS format - increasing
        atomic number with H at the end
        :param element_dict: dictionary keys = elements,
        values = atomic numbers - values can be discarded
        :return: list sorted into Tripos format
        """
        element_dict = dict((sorted(element_dict.items(),
                                    key=lambda item: item[1])))
        elements = [key for key in element_dict]

        # H needs to be at the end of the list of elements
        if 'H' in elements:
            elements.remove('H')
            elements.append('H')

        return elements

    def elements(self):
        """
        Returns a list of all the elements present in the input molecule,
        in increasing atomic number order with H last
        """
        element_dict = {}
        for atom in self.GetAtoms():
            if atom.GetSymbol() not in element_dict:
                element_dict[atom.GetSymbol()] = atom.GetAtomicNum()

        elements = self.sort_elements(element_dict)

        return elements

    def elements_by_index(self):
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

    @staticmethod
    def generate_sybyl_code(atom):
        """
        Generate the sybyl code for each atom - comprised of the symbol and
        atom type e.g. aromatic/sp3 etc.
        :param atom: instance of RDKit atom object in RDKit mol object
        :return: string of sybyl code for atom
        """
        symbol = atom.GetSymbol()

        if symbol == 'C' or symbol == 'N':
            if atom.GetIsAromatic():
                sybyl = f'{symbol}.ar'
            else:
                sybyl = f'{symbol}.{atom_types[str(atom.GetHybridization())]}'
        elif symbol == 'H':
            sybyl = 'H'
        else:
            sybyl = f'{symbol}.{atom_types[str(atom.GetHybridization())]}'

        return sybyl

    def get_atom_index_label(self, symbol, atom_index):
        """
        Generates the TRIPOS atom index label
        :param symbol: symbol of RDKit atom
        :param atom_index: index of atom in RDKit mol object
        :return: string = TRIPOS atom name
        """
        mol_by_elements = self.elements_by_index()

        return mol_by_elements[symbol].index(atom_index) + 1

    @staticmethod
    def get_atom_name(symbol, index_label):
        """
        Returns the TRIPOS atom name
        :param symbol: atom symbol
        :param index_label: TRIPOS atom index label
        :return: name of atom
        """
        return f'{symbol}{index_label}'

    def sort_atom_dataframe(self, dataframe):
        """
        Sorts TRIPOS atom dataframe by sybyl - elements first, followed by
        atom index label
        :param dataframe: pandas dataframe of TRIPOS atom information
        :return: pandas dataframe sorted by sybyl code
        """
        dataframe['atom_symbol'] = pd.Categorical(dataframe['atom_symbol'],
                                                  self.elements())

        dataframe = dataframe.sort_values(by=['atom_symbol',
                                              'atom_index_label'])
        dataframe.index = np.arange(1, len(dataframe) + 1)

        return dataframe

    @staticmethod
    def generate_index_mapper(dataframe):
        """
        TRIPOS Atom ID's and RDKit atom indices are not the same - a mapping is
        needed between the rdkit indices and the TRIPOS indices to generate
        the bond block.
        :param dataframe: pandas dataframe containing the information for
        the TRIPOS atom block
        :return: dictionary mapping rdkit atom ID's to TRIPOS atom ID's
        """
        index_mapper = pd.DataFrame(data=np.arange(1, len(dataframe) + 1),
                                    index=dataframe['rdkit_index'])

        index_lookup = index_mapper.to_dict()[0]

        return index_lookup

    def get_bond_dataframe(self):
        """
        Generates a pandas dataframe containing the information necessary to
        construct the TRIPOS bond block
        :return: pandas dataframe of bond indices (beginning and end),
        and TRIPOS bond type
        """

        tripos_bond = pd.DataFrame(
            columns=['begin_atom_rdkit', 'end_atom_rdkit', 'bond_type'])

        for index, bond in enumerate(self.GetBonds()):
            beginning = bond.GetBeginAtom().GetIdx()
            end = bond.GetEndAtom().GetIdx()

            # Get rdkit bond type and convert to TRIPOS bond type
            bond_type = str(bond.GetBondType())
            bond_type = bond_types[bond_type]

            tripos_bond = tripos_bond.append(
                {'begin_atom_rdkit': beginning, 'end_atom_rdkit': end,
                 'bond_type': bond_type}, ignore_index=True)

        return tripos_bond

    def sort_bond_dataframe(self, dataframe):
        """
        Converts RDKit indices in dataframe into TRIPOS atom ID numbers so
        atom IDs are consistent between atom and bond block. Then sorts by
        bond ID
        :param dataframe: pandas dataframe containing bond block information
        :return: pandas dataframe sorted by bond ID, with atom ID's replaced
        with TRIPOS ID's
        """
        # Replace beginning and end atom indices with TRIPOS atom indices
        # to match those in ATOM Block
        dataframe['begin'] = dataframe.apply(
            lambda row: self.index_lookup[row.begin_atom_rdkit], axis=1)
        dataframe['end'] = dataframe.apply(
            lambda row: self.index_lookup[row.end_atom_rdkit], axis=1)

        # Sort by bond type
        dataframe['bond_type'] = pd.Categorical(dataframe['bond_type'],
                                                list(bond_types.values()))

        tripos_bond = dataframe.sort_values(by=['bond_type', 'begin'])
        tripos_bond.index = np.arange(1, len(tripos_bond) + 1)
        tripos_bond = tripos_bond[['begin', 'end', 'bond_type']]

        return tripos_bond

    def get_xyz_coordinates(self):
        """
        Extract coordinates from .xyz file to assign to atoms instantiated
        from smiles string
        :return:list of lists of atomic symbol and x, y, z coordinates.
        Lists in order of atom index
        """
        xyz_file = self.xyz_file
        all_coords = self.open_file(xyz_file)

        # skip the first two lines of all_coords as it doesn't contain
        # coordinate information
        coordinates = [entry[0].split() for entry in all_coords[2:]]

        return coordinates

    @property
    def xyz_file(self):
        """
        Returns path to xyz file containing optimised geometrical coordinates
        :return:
        """
        return os.path.join('..', 'autodE_outputs', f'{self.name}.xyz')

    @staticmethod
    def get_coordinates_from_list(coords_list):
        """
        extracts the x,y,z coordinates from a list generated by
        self.get_xyz_coordinates
        :param coords_list: list of coords generated by
        self.get_xyz_coordinates
        :return: x, y and z co-ordinates
        """

        x = float(coords_list[1])
        y = float(coords_list[2])
        z = float(coords_list[3])

        return x, y, z
