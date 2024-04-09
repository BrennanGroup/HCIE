import numpy as np
import rdkit.Chem
import scipy
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D, Point2D

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
        self.name = name
        self.functionalisable_bonds = []
        self.shape_scores = {}
        self.esp_scores = {}
        self.total_scores = []
        self.alignment_vector = None

        if ("[R]" or "*") in self.smiles:
            self.smiles = self.smiles.replace("[R]", "[*]")
            self.rdmol = self.load_rdkit_mol_from_smiles()
            self.alignment_vector = self.get_alignment_vector_from_dummy_mol(self.rdmol)
            self.rdmol = self._replace_dummy_atom_with_hydrogen(self.rdmol)
        else:
            self.rdmol = self.load_rdkit_mol_from_smiles()

        self.embed_mol()

        self.charges = self._calculate_gasteiger_charges() if not charges else charges
        self.functionalisable_bonds = self.get_functionalisable_bonds()

        if len(self.functionalisable_bonds) > 1:
            # Each bond has two possible alignments (the original alignment and its mirror image) so 2 conformers
            # needed
            self.num_conformers = 2 * len(self.functionalisable_bonds)
        else:
            self.num_conformers = 2

        self.generate_conformers()

    def _calculate_gasteiger_charges(self):
        """
        Calculates the partial charges using the RDKit implementation of Gasteiger's algorithm
        Returns
        -------
        A list of partial charges indexed by atom index
        """
        try:
            charges = [
                atom.GetDoubleProp("_GasteigerCharge") for atom in self.rdmol.GetAtoms()
            ]
        except KeyError:
            AllChem.ComputeGasteigerCharges(self.rdmol)
            charges = [
                atom.GetDoubleProp("_GasteigerCharge") for atom in self.rdmol.GetAtoms()
            ]

        return charges

    @property
    def coords(self):
        return [
            self.get_xyz_from_mol(conf_id) for conf_id in range(self.num_conformers)
        ]

    @property
    def mol_block(self):
        return [
            Chem.MolToMolBlock(self.rdmol, confId=conf_id)
            for conf_id in range(self.num_conformers)
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
        Generates the number of conformers defined by num_conformers, appending these to the rdmol object
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

    @staticmethod
    def get_alignment_vector_from_dummy_mol(dummy_mol: rdkit.Chem.Mol):
        """
        Identifies the atom ID of the dummy atom and it's bonded partner, thus allowing the identification of the bond
        to be used as an alignment vector.
        Returns
        -------
        tuple of (non-H atom idx, dummy atom idx)
        """
        dummy_atom_id = bonded_atom_id = None

        for atom in dummy_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_atom_id = atom.GetIdx()
                break

        for bond in dummy_mol.GetBonds():
            if bond.GetBeginAtomIdx() == dummy_atom_id:
                bonded_atom_id = bond.GetEndAtomIdx()
                break
            elif bond.GetEndAtomIdx() == dummy_atom_id:
                bonded_atom_id = bond.GetBeginAtomIdx()
                break

        return bonded_atom_id, dummy_atom_id

    def _replace_dummy_atom_with_hydrogen(self, mol):
        """

        Parameters
        ----------
        mol: RDkit mol with dummy atom marked with *

        Returns
        -------
        rdkit mol with dummy atom replaced with hydrogen
        """
        rw_mol = Chem.RWMol(mol)
        dummy_atom_id = self.alignment_vector[1]
        rw_mol.ReplaceAtom(dummy_atom_id, Chem.Atom(1))

        no_dummy_mol = rw_mol.GetMol()
        Chem.SanitizeMol(no_dummy_mol)

        return no_dummy_mol

    def get_functionalisable_bonds(self) -> list[tuple[int, int]]:
        """
        Identifies the MedChem vectors in a molecule, and returns a list of the atom indices of these bonds.

        Here, MedChem vector referes to an X-H bond, where X is almost always C or N - a bond that can be functionalised
        in a MedChem project.

        Returns
        -------
        A list of tuples of the atom indices of the atoms in each X-H 'vector' bond in the order
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

    def get_atom_ids_of_ring_plane(
        self, functionalisable_bond: tuple[int, int]
    ) -> tuple[int, int, int, int]:
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
        tuple(int, int, int, int): the atom IDs of the three ring atoms used to define the plane in the
        order (H-atom, non-H atom, non-H neighbour 1, non-H neighbour 2)
        """

        probe_ring_atom = self.rdmol.GetAtomWithIdx(functionalisable_bond[0])

        neighbors = [
            neighbor.GetIdx()
            for neighbor in probe_ring_atom.GetNeighbors()
            if neighbor.GetSymbol() != "H"
        ]

        return (
            functionalisable_bond[1],
            functionalisable_bond[0],
            neighbors[0],
            neighbors[1],
        )

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
        reference_bond_idxs: tuple[int, int],
        probe_bond_idxs: tuple[int, int],
        probe_conf_id: int,
        reference_conf_id: int = 0,
    ):
        """

        Parameters
        ----------
        probe_molecule
        reference_molecule
        reference_bond_idxs
        probe_bond_idxs
        probe_conf_id
        reference_conf_id
        """
        self.esp_score = None
        self.shape_score = None
        self.probe_mol = probe_molecule
        self.ref_mol = reference_molecule
        self.ref_bond_idxs = reference_bond_idxs
        self.probe_bond_idxs = probe_bond_idxs
        self.probe_conf_id = probe_conf_id
        self.ref_conf_id = reference_conf_id

        self.rotation_matrix = None

    def align_score(self, similarity_metric="tanimoto") -> (float, float):
        """
        Align the probe molecule to the reference molecule, score the alignment, and then update the dictionary
        of the probe molecule with the score
        Parameters
        -------
        :similarity_metric: str - similarity metric to use for scoring ESP. Defaults to Tanimoto.
        Returns
        -------
        esp_score: esp similarity score of the alignment, normalised between 0 and 1. Scored using the specified metric
        shape_score: shape similarity score of the alignment, normalised between 0 and 1
        """
        self.align_bonds_and_rings()

        for conf in (self.probe_conf_id, self.probe_conf_id + 1):
            shape_score = self.get_shape_similarity(conf)
            self.probe_mol.shape_scores[conf] = shape_score

            esp_score = self.calculate_esp_similarity(
                probe_conf_id=conf, similarity_metric=similarity_metric
            )
            self.probe_mol.esp_scores[conf] = esp_score

            self.probe_mol.total_scores.append(esp_score + shape_score)

        return esp_score, shape_score

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

        non_h_atom_indices = (self.probe_bond_idxs[0], self.ref_bond_idxs[0])
        h_atom_indices = (self.probe_bond_idxs[1], self.ref_bond_idxs[1])

        rmsd = Chem.rdMolAlign.AlignMol(
            prbMol=self.probe_mol.rdmol,
            refMol=self.ref_mol.rdmol,
            prbCid=self.probe_conf_id,
            atomMap=[h_atom_indices, non_h_atom_indices],
        )

        return rmsd

    @staticmethod
    def translate_mol_to_origin(coords, atom_idx):
        """
        Translates non-h-atom in H-X bond to the origin. This is necessary for aligning molecules, as most rotations
        will be about the origin.
        Parameters
        -------

        Returns
        -------
        numpy array of coordinates for molecule.
        """
        return coords - coords[atom_idx]

    @staticmethod
    def rotate_about_bond(coords, axis, theta, origin=None):
        """
        Rotates a molecule about an axis by an angle theta. First the molecule is translated such that the ring atom is
        centred at the origin, then rotated about the axis defined by the X-H bond by a rotation matrix. It is then
        translated back from the origin.

        Parameters
        ----------
        coords: coordinates of molecule to rotate.
        axis: Vector defining the rotation. In almost all cases this will be an X-H bond
        theta: Angle (in radians) to rotate about.
        origin: Optional. Vector to translate molecule by. Is translated back after rotation.

        Returns
        -------
        numpy array of rotated coordinates.
        """
        if origin is not None:
            coords = np.asarray([coord - np.asarray(origin)] for coord in coords)

        # Ensure axis is cast as an array, and normalise it to unit length
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)

        # Compute the 3D rotation matrix using the Euler-Rodrigues formula
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_matrix = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

        rotated_coords = np.array([np.matmul(rot_matrix, coord) for coord in coords])

        if origin is not None:
            for coord in rotated_coords:
                coord += origin

        return rotated_coords

    def align_bonds_and_rings(self):
        """
        Aligns a probe molecule to a query (reference) molecule along a specified bond, and aligns the ring planes
        using Kabsch's algorithm to minimise the RMSD of the points.

        The alignment generated by this process does not take into account the C2 symmetry axis of the bond, nor does it
        maximise overlap for bicyclic ring systems. Therefore a 180 degree rotation about the aligned bond after the
        alignment is also necessary, and both should be scored.
        Returns
        -------

        """
        probe_atom_ids = self.probe_mol.get_atom_ids_of_ring_plane(self.probe_bond_idxs)
        reference_atom_ids = self.ref_mol.get_atom_ids_of_ring_plane(self.ref_bond_idxs)

        probe_coords = self.probe_mol.coords[self.probe_conf_id]
        reference_coords = self.ref_mol.coords[self.ref_conf_id]

        # In order to translate the probe molecule back onto the query molecule after alignment, a translation vector
        # that maps the probe to the reference is necessary
        translation_vector = (
            reference_coords[reference_atom_ids[1]] - probe_coords[probe_atom_ids[1]]
        )

        # Translate non-H atom of specified bond to the origin
        probe_coords_translated = self.translate_mol_to_origin(
            probe_coords, probe_atom_ids[1]
        )
        ref_coords_translated = self.translate_mol_to_origin(
            reference_coords, reference_atom_ids[1]
        )

        # Define the p and q matrices necessary for the alignment (p matrix is aligned onto q matrix)
        p_matrix = probe_coords_translated[probe_atom_ids, :]
        q_matrix = ref_coords_translated[reference_atom_ids, :]
        rotation_matrix = self.get_rot_mat_kabsch(p_matrix, q_matrix)

        # Apply the rotation to the probe molecule
        probe_coords_translated_aligned = self.rotate_by_matrix(
            probe_coords_translated, rotation_matrix
        )

        # Flip this by 180 degrees to generate the second alignment
        probe_coords_translated_aligned_flipped = self.rotate_about_bond(
            probe_coords_translated_aligned,
            probe_coords_translated_aligned[probe_atom_ids[0]]
            - probe_coords_translated_aligned[probe_atom_ids[1]],
            np.pi,
        )

        # Now translate back on top of query (reference) molecule
        probe_coords_aligned = (
            probe_coords_translated_aligned
            + probe_coords[probe_atom_ids[1]]
            + translation_vector
        )
        probe_coords_aligned_flipped = (
            probe_coords_translated_aligned_flipped
            + probe_coords[probe_atom_ids[1]]
            + translation_vector
        )

        # Update probe molecule coordinates
        self.probe_mol.update_rdkit_mol_coords(probe_coords_aligned, self.probe_conf_id)
        self.probe_mol.update_rdkit_mol_coords(
            probe_coords_aligned_flipped, self.probe_conf_id + 1
        )

        return None

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
            self.ref_bond_idxs[0],
            self.probe_bond_idxs[0],
        )

        # Firstly it is necessary to translate the two molecules to the origin
        ref_translated = self.translate_coords(
            ref_mol_coords, vector=ref_mol_coords[ref_non_h_atom_idx]
        )
        probe_translated = self.translate_coords(
            probe_mol_coords, vector=probe_mol_coords[probe_non_h_atom_idx]
        )

        # Now determine the rotation matrix using Kabsch's algorithm
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

        # Rotate molecule about the designated bond by 180 degrees and print these too
        vector = (
            rotated_probe_coords[self.probe_bond_idxs[1]]
            - rotated_probe_coords[self.probe_bond_idxs[0]]
        )
        rotated_flipped_probe_coords = self.rotate_about_bond(
            rotated_probe_coords, vector, np.pi, origin=None
        )

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

    def calculate_esp_similarity(
        self, probe_conf_id: int, similarity_metric="tanimoto", renormalise=True
    ):
        """
        Calculates the ESP similarity of the alignment and its mirror, and stores these as an attribute of the probe
        molecule
        Parameters
        -------
        :conf_id: id of the conformation to calculate the ESP similarity
        :similarity: similarity metric to use when calculating ESP similarity
        :renormalise: Whether to renormalise the ESP scores to lie in [0,1]. Defaults to True
        Returns
        -------

        """

        distance_probe_probe = scipy.spatial.distance.cdist(
            self.probe_mol.coords[probe_conf_id], self.probe_mol.coords[probe_conf_id]
        )
        distance_ref_ref = scipy.spatial.distance.cdist(
            self.ref_mol.coords[self.ref_conf_id], self.ref_mol.coords[self.ref_conf_id]
        )
        distance_probe_ref = scipy.spatial.distance.cdist(
            self.probe_mol.coords[probe_conf_id], self.ref_mol.coords[self.ref_conf_id]
        )

        int_probe_probe = self.calculate_gaussian_integrals(
            distance_probe_probe, self.probe_mol.charges, self.probe_mol.charges
        )

        int_ref_ref = self.calculate_gaussian_integrals(
            distance_ref_ref, self.ref_mol.charges, self.ref_mol.charges
        )

        int_probe_ref = self.calculate_gaussian_integrals(
            distance_probe_ref, self.probe_mol.charges, self.ref_mol.charges
        )

        similarity = self.calculate_similarity(
            int_probe_probe, int_ref_ref, int_probe_ref, metric=similarity_metric
        )

        if renormalise:
            similarity = self.renormalise_similarities(
                similarity, metric=similarity_metric
            )
            return similarity
        else:
            return similarity

    @staticmethod
    def calculate_similarity(
        int_probe_probe: float, int_ref_ref: float, int_probe_ref: float, metric
    ):
        """
        Calculates the similarity between overlap integral of the probe and the reference molecule, using the metric
        specified by metric
        Parameters
        ----------
        int_probe_probe: Self-overlap integral of probe molecule
        int_ref_ref: Self-overlap integral of reference molecule
        int_probe_ref: Overlap integral between probe and reference molecule
        metric: Metric to use for calculating similarity. Tanimoto or Carbo

        Returns
        -------
        Similarity between probe and reference
        """
        if metric == "tanimoto":
            numerator = int_probe_ref
            denominator = int_probe_probe + int_ref_ref - int_probe_ref
        elif metric == "carbo":
            numerator = int_probe_ref
            denominator = np.sqrt(int_probe_probe * int_ref_ref)
        else:
            raise ValueError("Unknown Similarity Metric")

        if denominator != 0:
            return numerator / denominator
        else:
            raise ValueError("Denominator in similarity calculation cannot be 0")

    @staticmethod
    def calculate_gaussian_integrals(
        distance, charges1: np.ndarray | list[float], charges2: np.ndarray | list[float]
    ):
        """
        Calculates the Gaussian overlap integrals for the coulombic charge, using 3 Gaussian functions to approximate
        the 1/r term, as described in DOI:10.1021/ci00007a002. The co-efficients are calculated by expanding out the
        overlap integral in terms of the Gaussians, and then calculating the standard integral and substituting in the
        width and height of the three Gaussian functions, and are taken without modification from
        https://doi.org/10.1021/acs.jcim.1c01535
        Parameters
        ----------
        distance: np.array matrix of atomic distances
        charges1: 1D array of partial charges of molecule 1
        charges2: 1D array of partial charges of molecule 2

        Returns
        -------
        Analytic overlap integral
        """
        # Pre-computed coefficients
        a = np.array(
            [
                [15.90600036, 3.9534831, 17.61453176],
                [3.9534831, 5.21580206, 1.91045387],
                [17.61453176, 1.91045387, 238.75820253],
            ]
        )
        b = np.array(
            [
                [-0.02495, -0.04539319, -0.00247124],
                [-0.04539319, -0.2513, -0.00258662],
                [-0.00247124, -0.00258662, -0.0013],
            ]
        )

        # Calculate pair-wise product of atomic charges, and then flatten.
        charges = (np.asarray(charges1)[:, None] * np.asarray(charges2)).flatten()

        distance = (distance**2).flatten()

        return (
            (a.flatten()[:, None] * np.exp(distance * b.flatten()[:, None])).sum(0)
            * charges
        ).sum()

    def get_shape_similarity(self, probe_conf_id):
        """
        Calculates the similarity of the shape between the reference molecule and the probe molecule
        Parameters
        ----------
        probe_conf_id: conformer id of probe molecule

        Returns
        -------
        Shape similarity score
        """

        return 1 - AllChem.ShapeTanimotoDist(
            self.probe_mol.rdmol,
            self.ref_mol.rdmol,
            confId1=probe_conf_id,
            confId2=self.ref_conf_id,
        )

    @staticmethod
    def renormalise_similarities(similarity: float, metric) -> float:
        """
        Renormalises the similarity score to between 0 and 1.
        Parameters
        ----------
        similarity: Similarity score to renormalise
        metric: similarity metric used to calculate original score (Tanimoto/carbo

        Returns
        -------
        Similarity: similarity score normalised to sit between 0 and 1
        """
        if metric == "tanimoto":
            return (similarity + 1 / 3) / (4 / 3)
        elif metric == "carbo":
            return (similarity + 1) / 2
        else:
            raise ValueError("Unknown similarity metric")

    @staticmethod
    def translate_coords(coords, vector):
        return coords - vector

    def alignments_to_sdf(self, filename=None):
        """
        Prints the reference molecule, and each of the aligned probe_mols to an sdf file, along with their ESP and shape
        similarity scores
        Parameters
        ----------
        filename: filename of SDF File

        Returns
        -------

        """
        if filename is None:
            filename = f"{self.ref_mol.name}_alignments.sdf"
        with Chem.SDWriter(filename) as w:

            self.ref_mol.rdmol.SetProp("_Name", f"{self.ref_mol.name}")
            w.write(self.ref_mol.rdmol, confId=self.ref_conf_id)

            for conf_id in range(self.probe_mol.num_conformers):
                self.probe_mol.rdmol.SetProp(
                    "_Name", f"{self.probe_mol.name}_{conf_id}"
                )
                self.probe_mol.rdmol.SetProp(
                    "ESP_sim", f"{self.probe_mol.esp_scores[conf_id]}"
                )
                self.probe_mol.rdmol.SetProp(
                    "Shape_sim", f"{self.probe_mol.shape_scores[conf_id]}"
                )
                w.write(self.probe_mol.rdmol, confId=conf_id)


if __name__ == "__main__":
    mol = Molecule("[R]C1=CC=CC2=CC=CC=C21", name="test")
    print(mol.alignment_vector)
    print(
        mol.alignment_vector[0],
        mol.rdmol.GetAtomWithIdx(mol.alignment_vector[0]).GetSymbol(),
    )
    print(
        mol.alignment_vector[1],
        mol.rdmol.GetAtomWithIdx(mol.alignment_vector[1]).GetSymbol(),
    )
