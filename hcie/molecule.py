import numpy as np
from itertools import combinations
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


class Molecule:
    def __init__(self,
                 smiles: str,
                 name: str = None):
        self.smiles = smiles.replace("[R]", "[*]")
        self.name = name if name is not None else 'query'
        self.user_defined_vectors = True if '[*]' in self.smiles else False
        self.user_vectors = None
        self.user_vector_hash = None
        self.aromaticity_flags = []

        self.shape_scores = {}
        self.esp_scores = {}
        self.total_scores = {}

        self.mol = self.generate_rdkit_mol()
        self.coords = self.xyz_from_mol()
        self.charges = self.calculate_gasteiger_charges()
        self.exit_vectors = self.get_exit_vectors()
        self.exit_vector_properties = self.calculate_exit_vector_measures()

        self.add_hashes_to_exit_vector_properties()

        self.exit_vector_properties_by_hash = self.get_exit_vector_properties_by_hash()

    @property
    def elements(self):
        """
        Returns a list of the elements in the query, where the position in the list is the atom idx of the atom
        :return: list of atomic symbols, where position in the list corresponds to the index of the atom
        """
        return [atom.GetSymbol() for atom in self.mol.GetAtoms()]

    @property
    def centroid(self):
        """
        The centroid of the molecule - this is calculated only from the aromatic atoms!
        :return: coordinates of the centroid of the molecule
        """
        aromatic_coords = [self.coords[idx] for idx, val in enumerate(self.aromaticity_flags) if val]
        return np.mean(aromatic_coords, axis=0)

    @property
    def num_exit_vectors(self) -> int:
        """
        Returns the number of exit vectors
        :return: number of exit vectors
        """
        return len(self.exit_vectors)

    @property
    def exit_vector_pairs(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Returns all the possible combinations of exit vector pairs in a query.
        :return: list of tuples of 2 tuples ((non-H atom, H-atom), (non-H atom, H-atom))
        """
        return list(combinations(self.exit_vectors, 2))

    def calculate_gasteiger_charges(self):
        """
        Calculates the partial charges using the RDKit implementation of Gasteiger's algorithm
        :return: A list of partial charges indexed by atom index
        """
        try:
            charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in self.mol.GetAtoms()]
        except KeyError:
            AllChem.ComputeGasteigerCharges(self.mol)
            charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in self.mol.GetAtoms()]
        return charges

    def get_coords(self, conf_id: int = 0):
        """
        Retrieves the coordinates for the specified conformer
        :param conf_id: ID of conformer to get coordinates for
        :return: np.ndarray of coordinates
        """
        if conf_id < 0 or conf_id >= self.mol.GetNumConformers():
            raise ValueError(f'Conformer ID {conf_id} is out of range')

        return self.xyz_from_mol(conf_id=conf_id)

    def generate_conformers(self, num_confs: int):
        """
        RDKit conformers are used to store the coordinates of the various alignments, and so there must be at least
        as many conformers for each query as there are going to be alignments generated - this means that there
        must be at least len(exit_vector_pairs) conformers.
        :param num_confs: Number of conformers to generate
        :return:
        """
        conf = Chem.Conformer(self.mol.GetConformer(0))

        for i in range(1, num_confs + 1):
            self.mol.AddConformer(conf, assignId=True)

        return None

    def generate_rdkit_mol(self):
        """
        Generates an RDKit query for the SMILES string provided by the user. If no exit_vectors are specified,
        then it is a simple case of instantiating, embedding, and optimising.
        If user-specified exit-vectors are provided, these need to be replaced with hydrogens before the query can be
        instantiated and embedded, but self.user_vectors is updated with the exit_vectors specified by the user
        :return: rdkit mol object, embedded
        """
        if not self.user_defined_vectors:
            mol = self.instantiate_and_embed_mol()
            return mol
        else:
            mol_with_dummies = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
            self.user_vectors = self.get_ids_of_user_vectors(mol_with_dummies)
            mol = self._replace_dummy_atom_with_hydrogen(mol_with_dummies)
            mol = self.embed_mol(mol)
            return mol

    def update_conformer_coords(self, new_coords: np.ndarray, conf_idx: int) -> None:
        """
        Updates the coordinates associated with conformer no. conf_idx to those specified in the np.ndarray
        new_coords. This is useful for storing the coordinates generated with an alignment in the requisite conformer.
        :param new_coords: np.ndarray of new coordinates
        :param conf_idx: conformer id to update
        :return: None
        """
        if len(new_coords) != self.mol.GetNumAtoms():
            raise ValueError('New coordinates do not have the same number of atoms as the query you are trying to '
                             'update')

        for idx in range(len(new_coords)):
            x, y, z = new_coords[idx][0], new_coords[idx][1], new_coords[idx][2]
            self.mol.GetConformer(conf_idx).SetAtomPosition(idx, Point3D(x, y, z))

        return None

    def _replace_dummy_atom_with_hydrogen(self, mol):
        """
        Replaces one or more dummy atoms in the query with a hydrogen atom
        :param mol: RDkit mol with dummy atom(s) marked with *
        :return: rdkit mol with dummy atom replaced with hydrogen
        """
        rw_mol = Chem.RWMol(mol)
        for user_vector in self.user_vectors:
            dummy_atom_id = user_vector[1]
            rw_mol.ReplaceAtom(dummy_atom_id, Chem.Atom(1))

        no_dummy_mol = rw_mol.GetMol()
        Chem.SanitizeMol(no_dummy_mol)

        return no_dummy_mol

    @staticmethod
    def get_ids_of_user_vectors(mol: rdkit.Chem.Mol) -> tuple:
        """
        Extracts the user specified exit vectors from an rdkit mol object, returning the vectors as a tuple of tuples
        of atom idx in the order (non-H atom, user-defined vector atom)
        :param mol: rdkit mol object with exit vectors indicated
        :return: tuple of user defined exit_vectors
        """
        user_vectors = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atoms have atomic number 0 in RDKit
                neighbour = atom.GetNeighbors()[0]
                user_vectors.append((neighbour.GetIdx(), atom.GetIdx()))

        return tuple(user_vectors)

    def instantiate_and_embed_mol(self) -> rdkit.Chem.Mol:
        """
        Instantiates an RDKit mol object from a SMILES string, and generates a 3D embedding.
        :return: RDKit mol object
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f'{self.smiles} is an invalid SMILES string')
        else:
            mol = Chem.AddHs(mol)

        mol = self.embed_mol(mol)

        return mol

    def embed_mol(self, mol: rdkit.Chem.Mol) -> rdkit.Chem.Mol:
        """
        Attempts to generate 3D coordinates by embedding an rdkit mol, and optimising it using MMFF
        :param mol: rdkit mol object to embed
        :return: embedded rdkit mol object
        """
        # Some of the weirder molecules need more attempts to embed, 100 000 attempts seems to catch them all
        if AllChem.EmbedMolecule(mol, maxAttempts=100000, randomSeed=42) != 0:
            raise RuntimeError('Molecule embedding failed')

        # The MMFF optimization step, which seems to generate reasonable geometries, messes the aromaticity flags,
        # so store the original ones from the SMILES string, and restore these after optimization
        self.aromaticity_flags = [atom.GetIsAromatic() for atom in mol.GetAtoms()]

        AllChem.MMFFOptimizeMolecule(mol)

        # Now restore original aromaticity flags
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetIsAromatic(self.aromaticity_flags[idx])

        return mol

    def xyz_from_mol(self, conf_id: int = 0) -> np.array:
        """
        Extract the XYZ coordinates from the mol.
        :param conf_id: conformer id to get coordinates from - defaults to 0 (the first conformer)
        :return: np.array of coordinates
        """
        xyz = np.zeros((self.mol.GetNumAtoms(), 3))
        conf = self.mol.GetConformer(conf_id)

        for i in range(conf.GetNumAtoms()):
            position = conf.GetAtomPosition(i)
            xyz[i, 0] = position.x
            xyz[i, 1] = position.y
            xyz[i, 2] = position.z

        return xyz

    def get_exit_vectors(self) -> list[tuple[int, int]]:
        """
        Identifies exit vectors (C-H or N-H) bonds in query
        :return: list of tuples of atom indices in each exit vector in the order (non-H atom index, H-atom index)
        """
        vector_atom_ids = []

        for bond in self.mol.GetBonds():
            if "H" in (bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()):
                begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()

                if end_atom.GetSymbol() == "H" and begin_atom.GetIsAromatic():
                    vector_atom_ids.append((begin_atom.GetIdx(), end_atom.GetIdx()))
                elif end_atom.GetIsAromatic():
                    vector_atom_ids.append((end_atom.GetIdx(), begin_atom.GetIdx()))
                else:
                    continue

        return vector_atom_ids

    def get_distance_between_atoms(self, atom1: int, atom2: int) -> float:
        """
        finds the distance (in Angstroms) between two atoms.
        :param atom1: idx of atom 1
        :param atom2: idx of atom 2
        :return: float - distance between the two atoms in angstroms
        """

        return float(np.linalg.norm(self.coords[atom1] - self.coords[atom2]))

    def get_angle_between_three_atoms(self, atom1: int, atom2: int, atom3: int) -> float:
        """
        Finds angle in degrees between three atoms
        :param atom1: idx of atom 1
        :param atom2: idx of atom 2
        :param atom3: idx of atom 3
        :return: float - angle in degrees between the three atoms
        """
        ba = self.coords[atom1] - self.coords[atom2]
        bc = self.coords[atom3] - self.coords[atom2]

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        return float(np.degrees(np.arccos(cosine_angle)))

    @staticmethod
    def angle_between_vectors(alpha_1: float, alpha_2: float) -> float:
        """
        Calculates the angle between two exit_vectors
        :param alpha_1: angle tail1-base1-base2 in degrees
        :param alpha_2: angle base1-base2-tail2 in degrees
        :return: angle between the vectors, in degrees
        """
        if alpha_1 > alpha_2:
            return float(np.abs(alpha_2 - (180 - alpha_1)))
        else:
            return float(np.abs(alpha_1 - (180 - alpha_2)))

    def get_measure_for_vector_pair(self, vector_pair: tuple[tuple[int, int], tuple[int, int]]) -> dict:
        """
        For a given vector pair defined by two pairs of atoms (a base (non-H) atom, and a tail (H) atom),
        calculate the distance between the base atoms, and the angle between (tail1-base1-base2) and (
        base1-base2-tail2) and return these as a dictionary.
        :param vector_pair:
        :return: dictionary
        """
        distance = self.get_distance_between_atoms(vector_pair[0][0], vector_pair[1][0])
        angles = self.calculate_angles_for_vector_pair(vector_pair)

        return {'vectors': vector_pair, 'distance': distance, 'angles': angles}

    def calculate_angles_for_vector_pair(self, vector_pair: tuple[tuple[int, int], tuple[int, int]]) -> dict:
        """
        Calculates the angles for a pair of exit vectors
        :param vector_pair:
        :return:
        """
        angle_t1b1b2 = self.get_angle_between_three_atoms(vector_pair[0][1], vector_pair[0][0], vector_pair[1][0])
        angle_b1b2t2 = self.get_angle_between_three_atoms(vector_pair[0][0], vector_pair[1][0], vector_pair[1][1])
        angle_between_vectors = self.angle_between_vectors(angle_t1b1b2, angle_b1b2t2)

        smaller_angle, larger_angle = sorted([angle_t1b1b2, angle_b1b2t2])

        return {'av': angle_between_vectors, 'a1': smaller_angle, 'a2': larger_angle}

    def calculate_exit_vector_measures(self) -> list[dict]:
        """
        Calculates the exit vector measure (distance, (av, a1, a2)) for each exit vector pair. Returns these as a
        list of dictionaries.
        :return: list of dictionaries of exit vector measures
        """
        return [self.get_measure_for_vector_pair(vector_pair) for vector_pair in self.exit_vector_pairs]

    @staticmethod
    def get_distance_hash(distance: float) -> str:
        """
        Calculates a 5 bit hash representing the bin that the distance between the two exit vectors is placed into.
        The righthand-most three digits represent the bin for the angle between the vectors - in this implementation
        there are 18 possible bins for the distance between two vectors:
        [0, 2), [2, 2.25), [2.25, 2.5), [2.5, 2.75), [2.75, 3.00), [3.00, 3.25), [3.25, 3.50), [3.50, 3.75), [3.75,
        4.00), [4.00, 4.25), [4.25, 4.50), [4.50, 4.75), [4.75, 5.00), [5.00, 5.25), [5.25, 5.50), [5.50, 5.75), [5.75,
        6.00), [6.00, inf)
        :param distance: float - distance between two exit vectors
        :return: 5 bit hash representing the distance between the two exit vectors
        """
        distance_bins = [0, 2, 2.25, 2.5, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75,
                         6.01, np.inf]

        distance_bin = np.digitize(distance, distance_bins, right=False) - 1

        return format(distance_bin, '05b')

    @staticmethod
    def get_angle_hash(angle_between_vectors: float) -> str:
        """
        Calculates a 3 bit hash representing the bin that the angle between the two exit vectors is placed into
        based on analysis of the angles between vectors in VEHICLe, the following bins were arrived at:
        [0, 10), [10, 25), [25, 85), [85, 135), [135, 165), [165, 180].
        :param angle_between_vectors: Angle between two exit vectors in degrees
        :return: 3 bit hash representing the angle between the two exit vectors
        """
        angle_bins = [0, 10, 25, 85, 135, 165, 180.1]
        angle_bin = np.digitize(angle_between_vectors, angle_bins, right=False) - 1

        return format(angle_bin, '03b')

    def add_hashes_to_exit_vector_properties(self) -> None:
        """
        Calculates the hashes for each of the exit vector pairs, and then adds this as a property to the
        exit_vector_properties dictionary
        :return:
        """
        for measure in self.exit_vector_properties:
            distance_hash = self.get_distance_hash(measure['distance'])
            angle_hash = self.get_angle_hash(measure['angles']['av'])
            measure['hash'] = distance_hash + angle_hash

            if measure['vectors'] == self.user_vectors:
                self.user_vector_hash = measure['hash']

        return None

    def get_exit_vector_properties_by_hash(self) -> dict:
        """
        Creates a dictionary with exit vector properties as values, and the hash as keys. As multiple vectors can
        correspond to the same hash, the output is a list of dictionaries, one for each exit vector
        :return: dictionary of list of dictionaries, with hashes as the keys
        """
        by_hash = defaultdict(list)

        for entry in self.exit_vector_properties:
            hash_key = entry['hash']

            properties_without_hash = {k: v for k, v in entry.items() if k != 'hash'}

            by_hash[hash_key].append(properties_without_hash)

        return dict(by_hash)

    def replace_hydrogens_with_dummy_atoms(self,
                                           atom_ids: list,
                                           update_mol: bool = False
    ) -> str | None:
        """
        Replaces hydrogens with dummy atoms, and then returns the SMILES string
        :param atom_ids: atom ids of atoms to replace
        :return:
        """
        rw_mol = Chem.RWMol(self.mol)
        for idx, atom in enumerate(atom_ids, start=1):
            dummy_atom = Chem.Atom(0)  
            if len(atom_ids) > 1:
                dummy_atom.SetAtomMapNum(idx)
            rw_mol.ReplaceAtom(atom, dummy_atom)

        dummy_mol = rw_mol.GetMol()
        Chem.SanitizeMol(dummy_mol)
        if update_mol:
            self.mol = dummy_mol
            return Chem.MolToSmiles(Chem.RemoveHs(dummy_mol))
        else:
            return Chem.MolToSmiles(Chem.RemoveHs(dummy_mol))

    def get_atom_ids_of_ring_plane(self,
                                   functionalisable_bond: tuple
                                   ) -> tuple:
        """
        Finds the atom indices of the non-H atoms bonded in a ring to the 'vector' bond. Used to define the p and q
        matrix for the rotation.

        The plane of the aromatic ring is here taken to be defined by three atoms within it. When rotating the probe
        molecule to make it coplanar with the reference molecule, the planes need to be defined.

        Parameters
        ----------
        functionalisable bond: tuple[int, int]: atom indices of atoms that define functionalisable bond in the order
        (non-H atom, H atom)

        Returns
        -------
        tuple(int, int, int): the atom IDs of the three ring atoms used to define the plane in the
        order (non-H atom, non-H neighbour 1, non-H neighbour 2)
        """
        probe_ring_atom = self.mol.GetAtomWithIdx(functionalisable_bond[0])

        neighbours = [
            neighbour.GetIdx()
            for neighbour in probe_ring_atom.GetNeighbors()
            if neighbour.GetSymbol() != "H"
        ]

        return (
            functionalisable_bond[0],
            neighbours[0],
            neighbours[1]
        )
