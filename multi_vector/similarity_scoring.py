"""
Module for storing all classes and functions for computing molecular similarity methods.
"""

import numpy as np
import scipy
from rdkit.Chem import AllChem

from constants import vdw_radii
from molecule import Molecule
from constants import gaussian_a_coefficients, gaussian_b_coefficients


def calculate_shape_similarity(probe_molecule: Molecule,
                               query_molecule: Molecule,
                               probe_conf_idx: int,
                               query_conf_idx: int) -> float:
    """
    Calculates the Tanimoto shape similarity of two molecules
    :param probe_molecule: probe molecule to score
    :param query_molecule: query molecule to score
    :param probe_conf_idx: conformer index of probe conformer to score against
    :param query_conf_idx: conformer index of query conformer to score against
    :return: Tanimoto shape similarity
    """
    return 1 - AllChem.ShapeTanimotoDist(
        probe_molecule.mol,
        query_molecule.mol,
        confId1=probe_conf_idx,
        confId2=query_conf_idx
    )


def calculate_gaussian_integrals(distance: np.ndarray,
                                 charges1: np.ndarray | list[float],
                                 charges2: np.ndarray | list[float]
                                 ) -> float:
    """
    Calculates the Gaussian overlap integrals for the coulombic charge, using 3 Gaussian functions to approximate
        the 1/r term, as described in DOI:10.1021/ci00007a002. The co-efficients are calculated by expanding out the
        overlap integral in terms of the Gaussians, and then calculating the standard integral and substituting in the
        width and height of the three Gaussian functions, and are taken without modification from
        https://doi.org/10.1021/acs.jcim.1c01535
    :param distance: numpy matrix of atomic distances
    :param charges1: 1D array of the atomic partial charges on molecule 1, indexed by atom index
    :param charges2: 1D array of the atomic partial charges on molecule 2, indexed by atom index
    :return: analytic overlap integral
    """
    # Calculate pair-wise product of atomic charges, and then flatten.
    charges = (np.asarray(charges1)[:, None] * np.asarray(charges2)).flatten()

    distance = (distance ** 2).flatten()

    return (
        (gaussian_a_coefficients.flatten()[:, None] * np.exp(distance * gaussian_b_coefficients.flatten()[:,
                                                                        None])).sum(0)
        * charges
    ).sum()


def calculate_distance_matrix(molecule_a: Molecule,
                              conf_idx_a: int,
                              molecule_b: Molecule = None,
                              conf_idx_b: int = None
                              ) -> np.ndarray:
    """
    Calculates the pairwise euclidean distance matrix for the coordinates of the molecule
    :param molecule_a: First molecule to calculate the distance matrix for
    :param conf_idx_a: Conformer index of the coordinates to use
    :param molecule_b: Optional: Second molecule to calculate the distance matrix for. If left blank the matrix is
    calculated for molecule_a's self distance matrix
    :param conf_idx_b: Conformer index of the coordinates to use in molecule b
    :return: distance matrix
    """
    if molecule_b is None:
        return scipy.spatial.distance.cdist(
            molecule_a.get_coords(conf_idx_a), molecule_a.get_coords(conf_idx_a), "euclidean"
        )
    else:
        return scipy.spatial.distance.cdist(
            molecule_a.get_coords(conf_idx_a), molecule_b.get_coords(conf_idx_b), "euclidean"
        )


def calculate_similarity(int_probe_probe: float,
                         int_query_query: float,
                         int_probe_query:float,
                         metric: str = "Tanimoto"
                         ) -> float:
    """
     Calculates the similarity between overlap integral of the probe and the reference molecule, using the metric
     specified by metric
    :param int_probe_probe: self-overlap integral of probe molecule
    :param int_query_query: self-overlap integral of query molecule
    :param int_probe_query: overlap integral between query and probe molecule
    :param metric: metric used to calculate similarity
    :return: similarity between probe and reference
    """
    if metric.casefold() == "Tanimoto".casefold():
        numerator = int_probe_query
        denominator = int_query_query + int_probe_probe - int_probe_query
    elif metric.casefold() == "Carbo".casefold():
        numerator = int_probe_query
        denominator = np.sqrt(int_query_query * int_probe_probe)
    else:
        raise ValueError("Unknown Similarity Metric")

    if denominator != 0:
        return (float(numerator / denominator) + 1/3) / (4/3)
    else:
        raise ValueError("Denominator in similarity calculation cannot be 0")


def calculate_esp_similarity(probe: Molecule,
                             query: Molecule,
                             probe_conf_idx: int,
                             query_conf_idx: int,
                             metric: str = "Tanimoto"
                             ) -> float:
    """
    Calculates the ESP similarity between the query and the probe molecule, using the method defined in DOI:10.1021/ci00007a002
    :param probe: probe molecule
    :param probe_conf_idx: conformer index of probe conformer to score against
    :param query: query molecule
    :param query_conf_idx: conformer index of query molecule to score against
    :param metric: similarity metric to use - defaults to Tanimoto
    :return: esp_similarity
    """
    distance_probe_probe = calculate_distance_matrix(probe, probe_conf_idx)
    distance_query_query = calculate_distance_matrix(query, query_conf_idx)
    distance_probe_query = calculate_distance_matrix(probe, probe_conf_idx, query, query_conf_idx)

    probe_self_integral = calculate_gaussian_integrals(distance_probe_probe, probe.charges, probe.charges)
    query_self_integral = calculate_gaussian_integrals(distance_query_query, query.charges, query.charges)
    probe_query_integral = calculate_gaussian_integrals(distance_probe_query, probe.charges, query.charges)

    return calculate_similarity(
        probe_self_integral, query_self_integral, probe_query_integral, metric
    )


class Grid:
    """
    A class for encoding the grid that is used to calculate the shape similarity of molecules. This is achieved by
    encoding the shape and volume of a query into a 3D grid that is made up of lots of cubic voxels. The atoms are
    positioned in the grid based on their atomic coordinates, and the number of voxels are 'coloured' based on the
    vdw radii of the element that each atom corresponds to. This gives a 3D representation of the volume of a
    query, but also gives a means of comparing their shapes, as if the two molecules have all the same voxels
    coloured, then they have a similarity of unity, otherwise the lower the intersection and the greater the union,
    the less similar the molecules are.
    """
    def __init__(self,
                 molecule: Molecule,
                 bottom_left: np.ndarray,
                 top_right: np.ndarray,
                 coords: np.ndarray = None,
                 step_size=0.1,
                 padding=2.0
                 ):
        """
        :param molecule: The query to encode.
        :param bottom_left: The bottom left coordinates of the query.
        :param top_right: The top right coordinates of the query.
        :param coords: numpy array of atomic coordinates
        :param query: instance of Molecule class corresponding to the coordinates provided.
        :param step_size: the width of the voxels in the grid. For now this defaults to 0.1 Angstroms
        :param padding: padding to put around the atoms at the extremities of the grid
        """
        self.molecule = molecule
        self.step_size = step_size
        self.padding = padding

        if coords is not None:
            self.coords = coords
        else:
            self.coords = self.molecule.coords

        self.min_coords = bottom_left - self.padding
        self.max_coords = top_right + self.padding

        self.shape = (int(np.ceil(self.x_length / self.step_size)),
                      int(np.ceil(self.y_length/self.step_size)),
                      int(np.ceil(self.z_length/self.step_size)))

        self.grid = np.zeros(self.shape, dtype=np.float32)

    @property
    def x_length(self):
        """
        The length of the x dimension of the grid
        :return: x length in Angstroms
        """
        return self.max_coords[0] - self.min_coords[0]

    @property
    def y_length(self):
        """
        The length of the y dimension of the grid
        :return: y length in Angstroms
        """
        return self.max_coords[1] - self.min_coords[1]

    @property
    def z_length(self):
        """
        The length of the z dimension of the grid
        :return: z length in Angstroms
        """
        return self.max_coords[2] - self.min_coords[2]

    def encode_mol_into_grid(self, ignore_hydrogens=False):
        """
        Encodes the shape of the aligned query into the grid, taking into account the vdw radii of the atoms.
        The logic is as follows:
        1. Determine the atom's position from its atomic coordinates
        2. Get the vdw radius from the dictionary.
        3. Convert the atomic coordinates in to a grid index.
        4. Determine the number of voxels to fill based on the vdw radius.
        5. Fill the appropriate number of voxels, in the correct position.
        :param ignore_hydrogens: if True, ignore hydrogen atoms.
        :return:
        """
        for idx, atom_coords in enumerate(self.coords):
            atom_voxel_index = self.get_voxel_index(atom_coords)
            atom_symbol = self.molecule.elements[idx]
            vdw_radius = vdw_radii[atom_symbol]  # This is in Angstroms

            if not (ignore_hydrogens and atom_symbol == 'H'):
                self.add_atom_to_grid(atom_voxel_index, vdw_radius)

        return None

    def add_atom_to_grid(self, atom_voxel_index: np.ndarray, vdw_radius: float):
        """
        Adds a new atom to the molecular grid
        :param atom_voxel_index: The index of the voxel representing the centre of the atom in the grid
        :param vdw_radius: The radius of the atom (in Angstroms)
        :return:
        """
        radius = vdw_radius / self.step_size  # This is now in voxels

        # Now loop over the 3D grid surrounding the voxel at the atom centre - this is complicated and a little
        # fiddly, but as yet this is the best way to do it
        x, y, z = atom_voxel_index[0], atom_voxel_index[1], atom_voxel_index[2]

        for i in range(max(0, int(x - radius)), min(self.shape[0], int(x + radius) + 1)):
            for j in range(max(0, int(y - radius)), min(self.shape[1], int(y + radius) + 1)):
                for k in range(max(0, int(z - radius)), min(self.shape[2], int(z + radius) + 1)):

                    # Calculate the Euclidean distance between the voxel at (i, j, k), and the voxel at the atom centre
                    distance = np.sqrt((i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2)
                    if distance <= radius:
                        self.grid[i, j, k] = 1

        return None

    def get_voxel_index(self, atom_coords: np.ndarray):
        """
        Takes the cartesian coordinates of an atom and returns the indices of the voxel coresponding to the atom's
        position
        :param atom_coords:
        :return:
        """
        return (atom_coords - self.min_coords)/self.step_size


class ShapeSimilarity:
    def __init__(self,
                 query_mol: Molecule,
                 probe_mol: Molecule,
                 query_coords: np.ndarray = None,
                 probe_coords: np.ndarray = None,
                 padding: float = 2.0):
        self.query_mol = query_mol
        self.probe_mol = probe_mol
        self.padding = padding
        self.query_coords = query_coords if query_coords is not None else None
        self.probe_coords = probe_coords if probe_coords is not None else None

        self.query_grid, self.probe_grid = self.embed_mols_in_grids()

    def get_union_box_dimensions(self):
        """
        Calculate the bottom left and top right corner of the smallest box large enough to contain both molecules (
        referred to here as the union box).
        :return: bottom left corner coords (np.ndarray), top right corner coords(np.ndarray)
        """
        probe_min_coords = np.min(self.probe_mol.coords, axis=0)
        probe_max_coords = np.max(self.probe_mol.coords, axis=0)

        query_min_coords = np.min(self.query_mol.coords, axis=0)
        query_max_coords = np.max(self.query_mol.coords, axis=0)

        union_box_minimum = np.minimum(probe_min_coords, query_min_coords)
        union_box_maximum = np.maximum(probe_max_coords, query_max_coords)

        return union_box_minimum, union_box_maximum

    def embed_mols_in_grids(self):
        """
        Calculates the tanimoto similarity score between the two molecules
        :return:
        """
        query_coords = self.query_coords if self.query_coords is not None else self.query_mol.coords
        probe_coords = self.probe_coords if self.probe_coords is not None else self.probe_mol.coords

        union_min, union_max = self.get_union_box_dimensions()

        query_grid = Grid(molecule=self.query_mol, bottom_left=union_min, top_right=union_max, coords=query_coords)
        probe_grid = Grid(molecule=self.probe_mol, bottom_left=union_min, top_right=union_max, coords=probe_coords)

        for grid in (query_grid, probe_grid):
            grid.encode_mol_into_grid()

        assert query_grid.shape == probe_grid.shape

        return query_grid, probe_grid

    def calculate_tanimoto_similarity(self) -> float:
        """
        Calculates the tanimoto similarity between the query and the probe in the instantiated alignment
        :return: Tanimoto Score
        """
        intersection = np.sum(np.minimum(self.query_grid.grid, self.probe_grid.grid))
        union = np.sum(np.maximum(self.query_grid.grid, self.probe_grid.grid))

        return intersection / union
