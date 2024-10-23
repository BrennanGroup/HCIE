"""
Module containing the classes and functions necessary for aligning molecules together
"""

import numpy as np

from hcie.molecule import Molecule
from hcie.similarity_scoring import calculate_shape_similarity, calculate_esp_similarity


class Alignment:
    def __init__(self,
                 probe_molecule: Molecule,
                 query_molecule: Molecule,
                 query_exit_vectors: list | tuple,
                 probe_exit_vectors: list | tuple,
                 probe_conformer_idx: int,
                 query_conformer_idx: int = 0
                 ):
        self.probe = probe_molecule
        self.query = query_molecule
        self.query_vectors = query_exit_vectors
        self.probe_vectors = probe_exit_vectors
        self.probe_conf_idx = probe_conformer_idx
        self.query_conf_idx = query_conformer_idx

    def update_probe_coords(self, new_coords: np.ndarray):
        """
        Update the probe query's coords with new coords. These will be stored at the conformer assocoated with
        self.probe_conformer_idx. This will most likely be called after alignment, and update the probe's coordinates to
        aligned coords.
        :param new_coords: The new coordinates to update the probe query with.
        :return: None
        """
        self.probe.update_conformer_coords(new_coords=new_coords, conf_idx=self.probe_conf_idx)
        return None

    @staticmethod
    def calc_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        Calculates the RMSD between two sets of coordinates without any further alignment.
        This will be for any atoms that are in the coords array, so could feasibly include H atoms.
        :param coords1: coords of first set of atoms shape = (n, 3)
        :param coords2: coords of second set of atoms, shape = (n, 3)
        :return: Root Mean Squared distance, in Angstroms.
        """
        if coords1.shape != coords2.shape:
            raise ValueError("Coordinates must be the same shape")

        # Translate coords so centroid is at origin - most likely this will already be the case so this bit should do
        # nothing
        coords1_centred = np.array(coords1, copy=True)
        coords1_centred -= np.average(coords1_centred, axis=0)

        coords2_centred = np.array(coords2, copy=True)
        coords2_centred -= np.average(coords2_centred, axis=0)

        return np.sqrt(np.average(np.square(coords2_centred - coords1_centred)))

    @staticmethod
    def rotate_by_matrix(coords: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Rotates object defined by coords by the specified matrix. The centre of rotation must be at the origin
        :param coords: Coordinates of object to be rotated
        :param rotation_matrix: Matrix defining the rotation
        :return: Coordinates of rotated object
        """
        return np.dot(rotation_matrix, coords.T).T

    @staticmethod
    def get_kabsch_rotation_matrix(probe_matrix: np.ndarray, query_matrix: np.ndarray) -> np.ndarray:
        """
        Get the optimal rotation matrix with the Kabsch algorithm.

        Notation is from https://en.wikipedia.org/wiki/Kabsch_algorithm. Finds the matrix for the rotation of minimal
        RMSD that rotates object defined by p_matrix onto object defined by q_matrix. Note that the two molecules
        must have the centre of rotation at the origin.

        :param probe_matrix: A matrix of the coordinates of the four atoms that define the two exit-vectors of
        alignment in the probe query
        :param query_matrix: A matrix of the coordinates of the four atoms that define the two exit-vectors of
        alignment in the query query
        :return: rotation matrix
        """
        covariance_matrix = np.matmul(probe_matrix.transpose(), query_matrix)
        u, _, vt = np.linalg.svd(covariance_matrix)
        determinant = np.linalg.det(np.matmul(vt.transpose(), u.transpose()))
        int_matrix = np.identity(3)
        int_matrix[2, 2] = determinant
        rotation_matrix = np.matmul(np.matmul(vt.transpose(), int_matrix), u.transpose())

        return rotation_matrix


class AlignmentTwoVector(Alignment):
    def __init__(
            self,
            probe_molecule: Molecule,
            query_molecule: Molecule,
            query_exit_vectors,
            probe_exit_vectors,
            probe_conformer_idx: int,
            query_conformer_idx: int = 0
    ):
        """
        Class for aligning a probe query to a query, specifically aligning the vectors specified by
        query_user_vectors to the vectors specified by probe_vectors - these have previously been found by a hash
        search.

        :param query_molecule: the instance of Molecule for the query. This is the user specified query.
        :param probe_molecule: the instance of Molecule for the probe molecule. This is the database query being
        compared to the user-specified query molecule.
        :param query_exit_vectors: The exit-vectors specified by the user as important for the alignment.
        :param probe_exit_vectors: The exit-vectors in the probe query that have been returned by the hash search.
        These have a similar geometry to the user-specified query exit-vectors, and so are likely to be a good match.
        :param probe_conformer_idx: the index of the conformer in the probe query to store the aligned coordinates.
        """
        super().__init__(probe_molecule, query_molecule, query_exit_vectors, probe_exit_vectors, probe_conformer_idx,
                         query_conformer_idx)

    def align_and_score(self):
        """
        Aligns the probe molecule to the query molecule, minimising the RMSD between the probe_vectors and the
        query_vectors. The coordinates of the probe query conformer specified by self.probe_conf_idx are then
        updated, and the alignment is scored.
        :return:
        """
        aligned_coords = self.align_probe_to_query()
        self.update_probe_coords(aligned_coords)
        shape_sim = calculate_shape_similarity(self.probe,
                                               self.query,
                                               probe_conf_idx=self.probe_conf_idx,
                                               query_conf_idx=0)
        esp_sim = calculate_esp_similarity(self.probe,
                                           self.query,
                                           probe_conf_idx=self.probe_conf_idx,
                                           query_conf_idx=0,
                                           metric="Tanimoto")
        return shape_sim, esp_sim

    def align_probe_to_query(self):
        """
        Align the query molecule to the probe molecule using Kabsch's algorithm.
        :return:
        """
        # Unpack the list of lists of exit-vector atom IDs into a 1D list
        query_vector_ids = [item for vector_list in self.query_vectors for item in vector_list]
        probe_vector_ids = [item for vector_list in self.probe_vectors for item in vector_list]

        # Translate both molecules to the origin
        query_centered = self.query.coords - self.query.centroid
        probe_centered = self.probe.coords - self.probe.centroid

        # Define the matrices necessary for the Kabsch alignment procedure (probe is aligned to query)
        probe_matrix = probe_centered[probe_vector_ids, :]
        query_matrix = query_centered[query_vector_ids, :]
        rotation_matrix = self.get_kabsch_rotation_matrix(probe_matrix, query_matrix)

        rotated_probe = self.rotate_by_matrix(probe_centered, rotation_matrix)

        translated_probe = rotated_probe + self.query.centroid

        #aligned_rmsd = self.calc_rmsd(rotated_probe[probe_vector_ids], query_centered[query_vector_ids])

        return translated_probe


class AlignmentOneVector(Alignment):
    def __init__(self,
                 probe_molecule: Molecule,
                 query_molecule: Molecule,
                 query_exit_vectors,
                 probe_exit_vectors,
                 probe_conformer_idx: int,
                 query_conformer_idx: int = 0
                 ):
        super().__init__(probe_molecule, query_molecule, query_exit_vectors, probe_exit_vectors, probe_conformer_idx,
                         query_conformer_idx)
        if len(self.query_vectors) != 2:
            raise ValueError('One Vector Alignments should only have one exit_vector specified')

    def align_and_score(self, similarity_metric="Tanimoto") -> None:
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

        for conf in (self.probe_conf_idx, self.probe_conf_idx + 1):
            shape_sim = calculate_shape_similarity(self.probe,
                                                   self.query,
                                                   probe_conf_idx=conf,
                                                   query_conf_idx=0)
            esp_sim = calculate_esp_similarity(self.probe,
                                               self.query,
                                               probe_conf_idx=conf,
                                               query_conf_idx=0,
                                               metric=similarity_metric)
            self.probe.shape_scores[conf] = shape_sim
            self.probe.esp_scores[conf] = esp_sim
            self.probe.total_scores[conf] = shape_sim + esp_sim

        return None

    def align_bonds_and_rings(self):
        """
        Aligns a probe molecule to a query (reference) molecule along a specified bond, and aligns the ring planes
        using Kabsch's algorithm to minimise the RMSD of the points.

        The alignment generated by this process does not take into account the C2 symmetry axis of the bond, nor does it
        maximise overlap for bicyclic ring systems. Therefore a 180 degree rotation about the aligned bond after the
        alignment is also necessary, and both should be scored.
        :return:
        """
        probe_atom_ids = self.probe.get_atom_ids_of_ring_plane(self.probe_vectors)
        query_atom_ids = self.query.get_atom_ids_of_ring_plane(self.query_vectors)

        probe_coords = self.probe.get_coords(self.probe_conf_idx)
        query_coords = self.query.get_coords(self.query_conf_idx)

        probe_plane_centroid = np.mean(probe_coords[probe_atom_ids, :], axis=0)
        query_plane_centroid = np.mean(query_coords[query_atom_ids, :], axis=0)

        # In order to translate the probe molecule back onto the query molecule after alignment, a translation vector
        # that maps the probe to the reference is necessary
        translation_vector = query_plane_centroid - probe_plane_centroid

        # Translate each molecule so that the centroid of the three atoms to be aligned is at the origin
        probe_coords_origin = probe_coords - probe_plane_centroid
        query_coords_origin = query_coords - query_plane_centroid

        for idx, flip in enumerate([False, True]):
            aligned_coords = self.apply_kabsch_alignment(
                probe_coords_origin,
                query_coords_origin,
                probe_atom_ids,
                query_atom_ids,
                flip=flip
            )
        # Translate the probe back on top of the query atom for scoring and update coords on Molecule
            aligned_coords += (probe_plane_centroid + translation_vector)
            self.probe.update_conformer_coords(aligned_coords, self.probe_conf_idx + idx)

        return None

    def apply_kabsch_alignment(self,
                               probe_coords: np.ndarray,
                               query_coords: np.ndarray,
                               probe_atom_ids: np.ndarray | list,
                               query_atom_ids: np.ndarray | list,
                               flip: bool = False) -> np.ndarray:
        """
        Helper function to perform Kabsch alignment on a set of probe coordinates and query coordinates. If flip is
        True then the coordinates of the probe atoms are flipped, which is equivalent to rotating the molecule by 180
        degrees about the axis of the XH bond.
        :param probe_coords:
        :param query_coords:
        :param probe_atom_ids:
        :param query_atom_ids:
        :param flip:
        :return:
        """
        if flip:
            probe_atom_ids = (probe_atom_ids[0], probe_atom_ids[2], probe_atom_ids[1])


        probe_matrix = probe_coords[probe_atom_ids, :]
        query_matrix = query_coords[query_atom_ids, :]

        # Get Kabsch Matrix
        rotation_matrix = self.get_kabsch_rotation_matrix(probe_matrix, query_matrix)

        return self.rotate_by_matrix(probe_coords, rotation_matrix)

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
        coords = np.asarray(coords)

        if origin is not None:
            origin = np.asarray(origin)
            coords = coords - origin

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
            rotated_coords += origin

        return rotated_coords
