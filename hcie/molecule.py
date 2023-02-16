"""
Contains functionality for Molecule class - inherits from RDKit Molecule class instantiated from a SMILES string
"""
from hcie.mol2_dict import bond_types, atom_types, NO_SUBSTRUCTS, MOLECULE_TYPE
from rdkit import Chem
import os
import shutil
import json
import csv
import autode
import subprocess
import pkg_resources
import xyz2mol

# Set path to package data files referenced within the code
VEHICLE_MOL2_FILENAME = pkg_resources.resource_filename("hcie", "Data/vehicle_dft.mol2")
PYMOL_VISUALISATION = pkg_resources.resource_filename('hcie', 'Data/pymol_visualisation.py')

with open(pkg_resources.resource_filename("hcie", "Data/vehicle_smiles.json"), 'r') as smiles_dict:
    vehicle_smiles = json.load(smiles_dict)


class Molecule(Chem.Mol):
    def __init__(
        self,
        arg: str,
        name: str = "query",
        mol2: str = None,
        output_queries: bool = True,
        max_hits: int = 20,
    ):
        """
        Molecule class. Contains information about the molecule necessary to construct mol2 files, optimise geometries,
        and perform VEHICLe searches
            :param arg: SMILES string of molecule to instantiate or path to xyz_file
            :param name: name of molecule, defaults to query
            :param mol2: mol2 file of molecule - if it exists, if not one will be created
            :param output_queries: A ShaEP parameter - Writes the matched query structure to the structures file before
            the superimposed structures
            :param max_hits: A ShaEP parameter - The maximum number of molecules to include in the hitlist, default is
            20, 0 imposes no limit.
        """
        self.avg_similarity_scores = None
        self.shape_similarity_scores = None
        self.esp_similarity_scores = None
        self.regid_hits = None
        self.smiles_hits = None
        self.xyz_filename = None
        self.coordinates: list = []
        self.charges: list = []
        self.optimise = True

        if arg is not None and arg.endswith(".xyz"):
            self.xyz_filename = arg
            self._init_from_xyz_file(arg)
            self.optimise = False
            if name == "query":
                self.name = self._get_name_from_xyz_file(arg, name)
        elif arg is not None:
            super().__init__(Chem.AddHs(Chem.MolFromSmiles(arg)))
            self.smiles = arg
            self.name = name

        self.mol2 = mol2

        # Attributes relative to ShaEP searches - unlikely to need to change these
        self.output_queries = output_queries
        self.max_hits = max_hits

    def __str__(self):
        return f"Molecule({self.name}, smiles={self.smiles})"

    def _init_from_xyz_file(self, xyz_filename: str):
        """
        Instantiates a molecule from xyz file - using xyztomol (https://github.com/jensengroup/xyz2mol)
        :param xyz_filename: filename of xyz file to instantiate from
        :return: RDKit mol object
        """
        atoms, charges, xyz_coordinates = xyz2mol.read_xyz_file(xyz_filename)
        mols_from_xyz_file = xyz2mol.xyz2mol(
            atoms=atoms,
            coordinates=xyz_coordinates,
            charge=charges,
            allow_charged_fragments=True,
            use_graph=True,
            use_huckel=True,
            embed_chiral=True,
        )
        self.coordinates = xyz_coordinates
        super().__init__(mols_from_xyz_file[0])
        os.remove('nul')
        os.remove('run.out')

        return None

    @staticmethod
    def _get_name_from_xyz_file(xyz_filename, name):
        """
        extracts the name from the second line of the xyz file - if empty then defaults to xyz filename.
        :param xyz_filename: filename of xyz file passed as argument
        :param name: name of molecule passed as argument - defaults to 'query'
        :return: xyz_filename
        """
        if name != 'query':
            return name
        with open(xyz_filename, 'r') as xyz_file:
            mol_name = xyz_file.readlines()[1].strip()
            if len(mol_name) > 1:
                return mol_name
            else:
                return os.path.basename(xyz_filename[:-4])

    @property
    def _atoms_by_tripos_order(self) -> list:
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

    @property
    def _rdkit_to_tripos_lookup(self) -> dict:
        """
        Generates a dictionary that maps the RDKit indices of all the atoms in the molecule to their TRIPOS ID's for the
        purposes of generating Mol2 files.
        :return: Dictionary - keys are RDKit indices, values are TRIPOS indices
        """
        lookup = {
            rdkit_idx: tripos_idx
            for tripos_idx, rdkit_idx in enumerate(self._atoms_by_tripos_order, 1)
        }

        return lookup

    def set_coordinates(self, mol: autode.Molecule) -> None:
        """
        Sets the coordinates attribute using optimised coordinates from the autode.Molecule instance
        :param mol: instance of autode.Molecule class, after geometry optimisation
        :return: list of lists of atomic coordinates, ordered by atom index
        """
        self.coordinates = mol.coordinates.tolist()

        return None

    @staticmethod
    def get_file_contents(filename) -> list:
        """
        opens file and returns a list of contents
        :param filename: path of file to open
        :return: list of file contents
        """
        with open(filename, "r") as file:
            csv_reader = csv.reader(file)
            return list(csv_reader)

    def optimise_geometry(self, method=autode.methods.XTB()) -> autode.Molecule:
        """
        Optimises the geometry and calculates the partial charges. Prints xyz file of optimised atomic co-ordinates
        :return: autode.Molecule
        """
        mol = autode.Molecule(name=self.name, smiles=self.smiles)
        mol.optimise(method=method)

        return mol

    def do_geometry_optimisation_and_set_charges_and_coordinates(
        self,
        optimise=True,
        optimise_method=autode.methods.XTB(),
        charge_method=autode.methods.XTB(),
        charge_keywords=autode.Config.XTB.keywords.opt,
    ) -> None:
        """
        Optimises the geometry of the instance of the Molecule, and sets the charges and coordinates after optimisation
        :param charge_keywords: keywords for electronic structure method to calculate charges
        :param charge_method: Electronic structure method to use for calculating charges
        :param optimise_method: Electronic structure method to use for optimising geometry
        :param optimise: Flag to indicate whether geometry optimisation should be performed on the molecule. If molecule
         is instantiated using xyz file then the geometry is assumed to be optimised and no optimisation is carried out.
        :return: None
        """
        if optimise:
            optimised_mol = self.optimise_geometry(method=optimise_method)
        else:
            optimised_mol = autode.Molecule(self.xyz_filename, name=self.name)
        self.set_charges(optimised_mol, method=charge_method, keywords=charge_keywords)
        self.set_coordinates(optimised_mol)

        return None

    def set_charges(
        self,
        mol: autode.Molecule = None,
        method=autode.methods.XTB(),
        keywords=autode.Config.XTB.keywords.opt,
    ) -> None:
        """
        Sets the charge attribute - optimise_geometry must have been run beforehand
        :param keywords: keywords to use for electronic structure method
        :param method: electronic structure method to use for calculating charges
        :param mol: autodE.Molecule
        :return: None
        """
        self.charges = self.calculate_charges(mol, method=method, keywords=keywords)

        return None

    def calculate_charges(
        self,
        molecule: autode.Molecule,
        method=autode.methods.XTB(),
        keywords=autode.Config.XTB.keywords.opt,
    ) -> list:
        """
        Calculate the atomic partial charges in the molecule using Mulliken population analysis
        :param keywords: keywords to use in electronic structure calculation, as defined in autode.Calculation
        :param method: Electronic structure method to use for calculation, as defined in autode.Calculation
        :param molecule: instance of autode.Molecule class to calculate partial charges
        :return: list of atomic partial charges, ordered by atomic index
        """

        calc = autode.Calculation(
            name=f"{self.name}_calc",
            method=method,
            keywords=keywords,
            molecule=molecule,
        )

        calc.run()
        atomic_charges = calc.get_atomic_charges()

        return atomic_charges

    @property
    def mol2_filename(self) -> str:
        """
        Path to mol2 file for instance of molecule
        :return: string
        """
        return f"{self.name}.mol2"

    @property
    def num_atoms(self) -> int:
        return len(self.GetAtoms())

    @property
    def num_bonds(self) -> int:
        return len(self.GetBonds())

    def write_mol2_molecule_block_to_file(self) -> None:
        """
        Writes <TRIPOS> molecule block to a Mol2 file, generating it if necessary, appending if exists.
        The number of features and sets are hard-coded to 0
        """
        with open(self.mol2_filename, "a") as mol2_file:
            print(
                f"@<TRIPOS>MOLECULE",
                f"{self.name}",
                f"{self.num_atoms} {self.num_bonds} {self.no_substructs} 0 0 ",
                f"{self.molecule_type}",
                f"{self.charge_type}",
                "",
                sep="\n",
                file=mol2_file,
            )

        return None

    @property
    def no_substructs(self) -> int:
        """
        Number of substructures in the molecule, for now hardcoded to 1
        :return: int 1
        """
        return NO_SUBSTRUCTS

    @property
    def molecule_type(self) -> str:
        """
        Molecule type
        :return: str - molecule type
        """
        return MOLECULE_TYPE

    @property
    def charge_type(self) -> str:
        """
        Type of charge used for mol2 file. Hardcoded to USERCHARGES
        :return: string
        """
        return "USER_CHARGES"

    def generate_sybyl_code(self, atom) -> str:
        """
        Generate the sybyl code for each atom - comprised of the symbol and
        atom type e.g. aromatic/sp3 etc.
        :param atom: instance of RDKit atom object in RDKit mol object
        :return: string of sybyl code for atom
        """
        symbol = atom.GetSymbol()

        if symbol == "C" or symbol == "N":
            if atom.GetIsAromatic():
                sybyl = f"{symbol}.ar"
            else:
                sybyl = f"{symbol}.{atom_types[str(atom.GetHybridization())]}"
        elif symbol == "H":
            sybyl = "H"
        else:
            sybyl = f"{symbol}.{atom_types[str(atom.GetHybridization())]}"

        return sybyl

    @staticmethod
    def get_atom_name(symbol, index_label) -> str:
        """
        Returns the TRIPOS atom name
        :param symbol: atom symbol
        :param index_label: TRIPOS atom index label
        :return: name of atom
        """
        return f"{symbol}{index_label}"

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
        if "H" in elements:
            elements.remove("H")
            elements.append("H")

        return elements

    def elements(self) -> list:
        """
        Returns a list of all the elements present in the input molecule,
        in increasing atomic number order with H last
        :return: list of element symbols
        """
        atomic_symbol_to_atomic_number = {}
        for atom in self.GetAtoms():
            if atom.GetSymbol() not in atomic_symbol_to_atomic_number:
                atomic_symbol_to_atomic_number[atom.GetSymbol()] = atom.GetAtomicNum()

        elements = self.sort_elements(atomic_symbol_to_atomic_number)

        return elements

    def elements_by_index(self) -> dict:
        """
        Returns a dictionary with element symbols as keys, and the atom ID's
        of those elements in the molecule as values
        """
        atom_ids_of_elements = {}
        for atom in self.GetAtoms():
            symbol = atom.GetSymbol()

            if symbol not in atom_ids_of_elements:
                atom_ids_of_elements[symbol] = []

            atom_ids_of_elements[symbol].append(atom.GetIdx())

        return atom_ids_of_elements

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
        with open(self.mol2_filename, "a") as mol2_file:
            print("@<TRIPOS>ATOM", file=mol2_file)
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
        substruct_name = "****"
        charge = self.charges[atom.GetIdx()]

        row = (
            f"{idx:<8}{atom_name:<11}{x:>7.3f}{y:>10.3f}{z:>10.3f} {sybyl:<10} "
            f"{substruct_no} {substruct_name:<11}{charge:>7.4f} "
        )

        return row

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
        with open(self.mol2_filename, "a") as mol2_file:
            print("@<TRIPOS>BOND", file=mol2_file)
            for tripos_bond_idx, bond in enumerate(self.GetBonds(), 1):
                origin_atom_id = self._rdkit_to_tripos_lookup[
                    bond.GetBeginAtom().GetIdx()
                ]
                target_atom_id = self._rdkit_to_tripos_lookup[
                    bond.GetEndAtom().GetIdx()
                ]
                bond_type = bond_types[str(bond.GetBondType())]

                print(
                    f"{tripos_bond_idx:<6}{origin_atom_id:<5}{target_atom_id:<5} {bond_type}",
                    file=mol2_file,
                )

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
        with open(self.mol2_filename, "a") as mol2_file:
            print("@<TRIPOS>SUBSTRUCTURE", file=mol2_file)
            print(
                "     1 ****        1 GROUP             0       ****    0 ROOT",
                file=mol2_file,
            )

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
            subprocess.run(
                [
                    "shaep",
                    "--maxhits",
                    str(self.max_hits),
                    "-q",
                    self.mol2_filename,
                    "--output-file",
                    "similarity.txt",
                    "--structures",
                    "overlay.sdf",
                    "--outputQuery",
                    VEHICLE_MOL2_FILENAME,
                ]
            )
        else:
            subprocess.run(
                [
                    "shaep",
                    "--maxhits",
                    str(self.max_hits),
                    "-q",
                    self.mol2_filename,
                    "--output-file",
                    "similarity.txt",
                    "--structures",
                    "overlay.sdf",
                    VEHICLE_MOL2_FILENAME,
                ]
            )

        return None

    def get_scores_from_similarity_file(self, file_path=None):
        """
        Reads in the similarity.txt file generated by shaep and sets the following attributes:
        regid_hits - the regid of the VEHICLe molecules, returned in order of increasing similarity
        smiles_hits - the smiles strings of the VEHICLe molecules, returned in order of increasing similarity
        avg_similarity_scores - the average of the shape similarity and esp similarity for each molecule, returned in
                                order of increasing similarity.
        shape_similarity_scores -  the molecular shape similarity score for each molecule, in order of increasing
                                    similarity
        esp_similarity_scores - the molecular electrostatics similarity score for each molecule, in order of increasing
                                similarity

        These scores are as defined in https://doi.org/10.1021/ci800315d
        :return: None
        """

        filepath = 'similarity_hits.txt' if file_path is None else file_path

        with open(filepath, 'r') as similarity_hits:
            similarity_hits_file_lines = similarity_hits.readlines()

        self.regid_hits = [line.split()[0] for line in similarity_hits_file_lines[1:]]
        self.smiles_hits = [vehicle_smiles[regid] for regid in self.regid_hits]
        self.avg_similarity_scores = [float(line.split()[1]) for line in similarity_hits_file_lines[1:]]
        self.shape_similarity_scores = [float(line.split()[2]) for line in similarity_hits_file_lines[1:]]
        self.esp_similarity_scores = [float(line.split()[3]) for line in similarity_hits_file_lines[1:]]

        return None

    def print_output_file(self, filename=None):
        """
        prints the output file for the HCIE search.
        :param filename: name for output file - defaults to self.name_output.csv
        :return: None
        """
        assert self.regid_hits is not None, 'similarity_hits.txt must be read'

        file_name = f'{self.name}_output.csv' if filename is None else filename

        with open(file_name, 'w') as output_file:
            print('regid', 'smiles', 'average_similarity', 'shape_similarity', 'esp_similarity',
                  sep=',', file=output_file)

            for idx in range(len(self.regid_hits)):
                print(self.regid_hits[idx],
                      self.smiles_hits[idx],
                      self.avg_similarity_scores[idx],
                      self.shape_similarity_scores[idx],
                      self.esp_similarity_scores[idx],
                      sep=',',
                      file=output_file)

        return None

    def shaep(self):
        """
        optimises the geometry using autodE, creates a mol2 file, and runs a shaep search of the molecule instance
        :return: None
        """
        self.do_geometry_optimisation_and_set_charges_and_coordinates(
            optimise=self.optimise
        )
        self.write_mol2_file()
        self.search_shaep()
        self.get_scores_from_similarity_file()
        self.print_output_file()

        shutil.copy(PYMOL_VISUALISATION, '.')

        return None
