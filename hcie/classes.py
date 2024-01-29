import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D, Point2D
from espsim import GetEspSim, GetShapeSim

RDLogger.DisableLog("rdApp.*")


class Molecule:
    def __init__(self, smiles: str, charges: list = None, name: str = None):
        """
        A basic representation of a molecule.

        Each instance has a SMILES string, and a number of num_conformers, each with a set of coordinates and charges.
        Parameters
        ----------
        smiles: standard SMILES string of the molecule.
        charges: list of partial charges, in order of atom index - optional
        name: name of molecule - optional
        """
        self.smiles = smiles
        self.charges = [] if not charges else charges
        self.name = name
        self.functionalisable_bonds = []

        self.rdmol = self.load_rdkit_mol_from_smiles()
        self.embed_mol()

        self.functionalisable_bonds = self.get_functionalisable_bonds()

        if len(self.functionalisable_bonds) > 1:
            self.num_conformers = len(self.functionalisable_bonds)
        else:
            self.num_conformers = 1

        self.generate_conformers()

    @property
    def coords(self):
        return [
            self.get_xyz_from_mol(conf_id) for conf_id in range(self.num_conformers)
        ]

    @property
    def mol_block(self):
        return [
            Chem.MolToMolBlock(self.rdmol, confId = conf_id) for conf_id in range(self.num_conformers)
        ]

    def embed_mol(self) -> None:
        """
        Embeds 3D coordinates of rdmol.

        Returns
        -------
        None
        """
        AllChem.EmbedMolecule(self.rdmol, maxAttempts=100000)

        return None

    def generate_conformers(self) -> None:
        """
        Generates the number of num_conformers defined by num_conformers, appending these to the rdmol object
        Returns
        -------
        None
        """

        conf = Chem.Conformer(self.rdmol.GetConformer(0))

        for i in range(1, self.num_conformers + 1):
            self.rdmol.AddConformer(conf, assignId=True)

        return None

    def load_rdkit_mol_from_smiles(self):
        """
        Generates an RDKit Mol from the smiles string, adding Hs if necessary.

        Returns
        -------
        RDKit Mol object
        """
        return Chem.AddHs(Chem.MolFromSmiles(self.smiles))

    def get_functionalisable_bonds(self):
        """
        Identifies the MedChem vectors in a molecule, and returns a list of the atom indices of these bonds.

        Here, MedChem vector referes to an X-H bond, where X is almost always C or N - a bond that can be functionalised
        in a MedChem project.

        Returns
        -------
        (list(tuple(int, int))): A list of tuples of the atom indices of the atoms in each X-H 'vector' bond in the order
                                (non-H atom index, H atom index).
        """
        vector_atom_ids = []

        for (
            idx,
            bond,
        ) in enumerate(self.rdmol.GetBonds()):
            if "H" in (bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()):
                begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()

                if end_atom.GetSymbol() == "H":
                    vector_atom_ids.append((begin_atom.GetIdx(), end_atom.GetIdx()))
                else:
                    vector_atom_ids.append((end_atom.GetIdx(), begin_atom.GetIdx()))

        return vector_atom_ids

    def update_rdkit_mol_coords(self, new_coords: np.array, conf_id: int):
        """
        Updates the coordinates of the RDKit mol representation of the molecule (conformer = conf_id) to new_coords
        Parameters
        ----------
        new_coords: np.array of new coordinates, indexed by atom index
        conf_id: Integer. Conformer to update.

        Returns
        -------
        None
        """

        for idx in range(len(new_coords)):
            x, y, z = new_coords[idx][0], new_coords[idx][1], new_coords[idx][2]
            self.rdmol.GetConformer(conf_id).SetAtomPosition(idx, Point3D(x, y, z))

        return None

    def get_xyz_from_mol(self, conf_id: int = 0):
        """
        Extracts a numpy array of coordinates from a molecules.

        Returns
        -------
        Numpy ndarray of shape `(N, 3)` where `N = mol.GetNumAtoms()`.
        """
        xyz = np.zeros((self.rdmol.GetNumAtoms(), 3))
        conf = self.rdmol.GetConformer(conf_id)

        for i in range(conf.GetNumAtoms()):
            position = conf.GetAtomPosition(i)
            xyz[i, 0] = position.x
            xyz[i, 1] = position.y
            xyz[i, 2] = position.z

        return xyz

    def get_atom_ids_of_ring_plane(self, non_h_atom_idx: int) -> tuple[int, int, int]:
        """
        Finds the atom indices of the non-H atoms bonded in a ring to the 'vector' bond. Used to define the plane of
        the ring.

        The plane of the aromatic ring is here taken to be defined by three atoms within it. When rotating the probe
        molecule to make it coplanar with the reference molecule, the planes need to be defined.

        Parameters
        ----------
        non_h_atom_idx: atom ID of non-H atom in X-H 'vector' bond used in alignment (int).

        Returns ------- tuple(int, int, int): the atom IDs of the three ring atoms used to define the plane. non-H
        X-H 'vector' atom is the middle entry.
        """

        probe_ring_atom = self.rdmol.GetAtomWithIdx(non_h_atom_idx)

        neighbors = [
            neighbor.GetIdx()
            for neighbor in probe_ring_atom.GetNeighbors()
            if neighbor.GetSymbol() != "H"
        ]

        return neighbors[0], non_h_atom_idx, neighbors[1]

    def translate(self, vector: np.array, conf_id: int) -> np.array:
        """
        Translate the coordinates of conformer conf_id by vector and update the stored coordinates
        Parameters
        ----------
        vector: np.array - vector to translate conformer by
        conf_id: integer, id of conformer
        Returns
        -------
        np.array - translated coordinates
        """
        new_coords = self.coords[conf_id] - vector

        self.update_rdkit_mol_coords(new_coords=new_coords, conf_id=conf_id)

        return new_coords

    def write_vectors_to_image(self, filename: str = None):
        """
        Generates a PNG file of the molecule with atom IDs, and the vector list. Useful for identifying the ID of the
        vector of importance
        Parameters
        ----------
        filename: filename to save the image to.

        Returns
        -------
        None
        """
        filename = f"{self.name}.png" if filename is None else filename

        canvas = rdMolDraw2D.MolDraw2DCairo(350, 300)
        canvas.drawOptions().addAtomIndices = True
        canvas.DrawMolecule(self.rdmol)
        canvas.SetFontSize(15)
        canvas.DrawString(f"{self.functionalisable_bonds}", Point2D(-3, -2.8), align=1)
        canvas.FinishDrawing()

        canvas.WriteDrawingText(filename)

        return None


class Alignment:
    def __init__(
        self,
        probe_molecule: Molecule,
        reference_molecule: Molecule,
        reference_bond_ids: tuple[int, int],
        probe_bond_ids: tuple[int, int],
        probe_conf_id: int,
        reference_conf_id: int = 0,
    ):
        """

        Parameters
        ----------
        probe_molecule
        reference_molecule
        reference_bond_ids
        probe_bond_ids
        probe_conf_id
        reference_conf_id
        """
        self.esp_score = None
        self.shape_score = None
        self.probe_mol = probe_molecule
        self.ref_mol = reference_molecule
        self.ref_bond_ids = reference_bond_ids
        self.probe_bond_ids = probe_bond_ids
        self.probe_conf_id = probe_conf_id
        self.ref_conf_id = reference_conf_id

        self.rotation_matrix = None

    def align_rotate_score(self):
        """
        Align the probe molecule to the reference molecule along the specified MedChem bond, rotate so that the
        aromatic rings are coplanar, and score with ESPSim
        Returns
        -------
        espsim score - float
        """
        self.align_mols_to_vectors()
        self.align_ring_planes()

        return self.score_alignment()

    def align_mols_to_vectors(self) -> float:
        """
        Aligns a probe molecule to a reference molecule along a specified X-H 'vector' bond.

        The aligned coordinates of the probe mol are updated in situ.

        This will align using rdkit rd MolAlign algorithm, so the alignment along the specified bond will be very
        good. For planar molecules the ring will often need rotating to make the rings coplanar.


        Returns
        -------
        (float): RMSD value of the alignment for reference.
        """

        non_h_atom_indices = (self.probe_bond_ids[0], self.ref_bond_ids[0])
        h_atom_indices = (self.probe_bond_ids[1], self.ref_bond_ids[1])

        rmsd = Chem.rdMolAlign.AlignMol(
            prbMol=self.probe_mol.rdmol,
            refMol=self.ref_mol.rdmol,
            prbCid=self.probe_conf_id,
            atomMap=[h_atom_indices, non_h_atom_indices],
        )

        return rmsd

    def align_ring_planes(self):
        """
        After the probe has been aligned to the reference along the specified MedChem vector bond, align the planes
        of the aromatic rings by rotating about an axis defined by the aligned bond, and update the coordinates.

        Returns
        -------
        None
        """
        ref_mol_coords = self.ref_mol.coords[self.ref_conf_id]
        probe_mol_coords = self.probe_mol.coords[self.probe_conf_id]

        ref_non_h_atom_idx, probe_non_h_atom_idx = (
            self.ref_bond_ids[0],
            self.probe_bond_ids[0],
        )

        # Firstly it is necessary to translate the two molecules to the origin
        ref_translated = self.translate_coords(
            ref_mol_coords, vector=ref_mol_coords[ref_non_h_atom_idx]
        )
        probe_translated = self.translate_coords(
            probe_mol_coords, vector=probe_mol_coords[probe_non_h_atom_idx]
        )

        # Now determine the rotation matrix
        p_matrix = self.get_plane_coords(
            probe_translated,
            self.probe_mol.get_atom_ids_of_ring_plane(probe_non_h_atom_idx),
        )
        q_matrix = self.get_plane_coords(
            ref_translated, self.ref_mol.get_atom_ids_of_ring_plane(ref_non_h_atom_idx)
        )
        rotation_matrix = self.get_rot_mat_kabsch(p_matrix, q_matrix)
        self.rotation_matrix = rotation_matrix

        # Rotate the translated probe molecule
        rotated_probe_coords = self.rotate_by_matrix(probe_translated, rotation_matrix)

        # Translate the rotated probe molecule back to its original position in space
        aligned_probe_coords = self.translate_coords(
            rotated_probe_coords, -probe_mol_coords[probe_non_h_atom_idx]
        )

        self.probe_mol.update_rdkit_mol_coords(
            aligned_probe_coords, conf_id=self.probe_conf_id
        )

        return None

    @staticmethod
    def get_rot_mat_kabsch(p_matrix: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
        """
        Get the optimal rotation matrix with the Kabsch algorithm.

        Notation is from https://en.wikipedia.org/wiki/Kabsch_algorithm. Finds the matrix for the rotation of minimal
        RMSD that rotates object defined by p_matrix onto object defined by q_matrix. Note that the two molecules
        must have the centre of rotation at the origin.

        --------------------------------------------------------------------------- Arguments: p_matrix (np.ndarray):
        A matrix of the coordinates of the three atoms defining the plane in the probe molecule ring - the aligned
        vector ring atom must be at the origin. q_matrix (np.ndarray): A matrix of coordinates of the three atoms
        defining the plane in the reference molecule ring - the aligned vector ring atom must be at the origin.

        Returns:
            (np.ndarray): rotation matrix
        """

        h = np.matmul(p_matrix.transpose(), q_matrix)
        u, _, vh = np.linalg.svd(h)
        d = np.linalg.det(np.matmul(vh.transpose(), u.transpose()))
        int_mat = np.identity(3)
        int_mat[2, 2] = d
        rot_matrix = np.matmul(np.matmul(vh.transpose(), int_mat), u.transpose())

        return rot_matrix

    @staticmethod
    def rotate_by_matrix(coords: np.array, rotation_matrix: np.array) -> np.array:
        """
        Rotates object defined by coords by matrix. Centre of rotation must be at the origin.

        Parameters
        ----------
        coords (np.array): coordinates of object to rotate.
        matrix (np.array): rotation matrix to apply.

        Returns
        -------
        np.array: coordinates of object rotated by matrix.
        """
        return np.dot(rotation_matrix, coords.T).T

    @staticmethod
    def get_plane_coords(coords: np.array, plane_atom_ids: tuple):
        """
        Returns the co-ordinates of the three atoms in the ring defining the plane used for the rotation.

        Used as the p and q matrix inputs for defining the Kabsch rotation matrix.
        Parameters
        ----------
        coords: np.array of coordinates for a molecule, indexed by atom id
        plane_atom_ids: atom ID's of the atoms that are being used to define the plane of the ring.

        Returns
        -------
        np.array of co-ordinates for the plane atoms
        """

        return coords[plane_atom_ids, :]

    def score_alignment(self):
        """
        Generate an ESPSim score for the alignment of the probe molecule onto the reference molecule
        Returns
        -------
        float - ESPSim score of electrostatic similarity
        float - ESP score of shape similarity
        """
        esp_sim = GetEspSim(
            prbMol=self.probe_mol.rdmol,
            refMol=self.ref_mol.rdmol,
            prbCid=self.probe_conf_id,
            renormalize=True,
        )

        shape_sim = GetShapeSim(
            prbMol=self.probe_mol.rdmol,
            refMol=self.ref_mol.rdmol,
            prbCid=self.probe_conf_id,
        )

        self.esp_score = esp_sim
        self.shape_score = shape_sim

        return esp_sim, shape_sim

    @staticmethod
    def translate_coords(coords, vector):
        return coords - vector


if __name__ == "__main__":
    quinoline = Molecule('c2ccc1ncccc1c2')
    indazole = Molecule('c2ccc1[nH]ncc1c2')

    #print(indazole.mol_block[0])
    #print(quinoline.mol_block[0])
    probe_bond_ids = [(2,11), (1, 10), (0, 9), (8, 14), (6, 13), (4, 12)]

    for conf_id, probe_bond in enumerate(probe_bond_ids):
        alignment = Alignment(indazole, quinoline, reference_bond_ids=(2, 12), probe_bond_ids=probe_bond, probe_conf_id=conf_id)
        alignment.align_rotate_score()
        print(alignment.esp_score)

    #print(indazole.mol_block[0])
    #print(quinoline.mol_block[0])