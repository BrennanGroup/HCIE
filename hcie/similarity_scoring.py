"""
Module for storing all classes and functions for computing molecular similarity methods.
"""

import numpy as np
import scipy
from rdkit.Chem import AllChem

from hcie.constants import vdw_radii
from hcie.molecule import Molecule
from hcie.constants import gaussian_a_coefficients, gaussian_b_coefficients


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
                              molecule_a_atoms: list = None,
                              molecule_b: Molecule = None,
                              conf_idx_b: int = None,
                              molecule_b_atoms: list = None
                              ) -> np.ndarray:
    """
    Calculates the pairwise Euclidean distance matrix for the coordinates of the molecule
    :param molecule_a: First molecule to calculate the distance matrix for
    :param conf_idx_a: Conformer index of the coordinates to use
    :param molecule_a_atoms: The atoms in molecule a to use for calculating the distance matrix - typically this will
    exclude non-aromatic hydrogen atoms
    :param molecule_b: Optional: Second molecule to calculate the distance matrix for. If left blank the matrix is
    calculated for molecule_a's self distance matrix
    :param conf_idx_b: Conformer index of the coordinates to use in molecule b
    :param molecule_b_atoms: The atoms in molecule b to use for calculating the distance matrix - typically this will
    exclude non-aromatic hydrogen atoms
    :return: distance matrix
    """
    coords_a = molecule_a.get_coords(conf_idx_a)
    coords_b = molecule_b.get_coords(conf_idx_b) if molecule_b is not None else coords_a

    coords_a = np.asarray(coords_a[molecule_a_atoms]) if molecule_a_atoms is not None else coords_a
    coords_b = np.asarray(coords_b[molecule_b_atoms]) if molecule_b_atoms is not None else coords_a

    return scipy.spatial.distance.cdist(coords_a, coords_b, 'euclidean')


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
    probe_atoms = probe.get_atoms_for_esp_calc()
    query_atoms = query.get_atoms_for_esp_calc()

    distance_probe_probe = calculate_distance_matrix(probe, probe_conf_idx, probe_atoms)
    distance_query_query = calculate_distance_matrix(query, query_conf_idx, query_atoms)
    distance_probe_query = calculate_distance_matrix(probe, probe_conf_idx, probe_atoms, query, query_conf_idx, query_atoms)

    probe_esp_charges = [probe.charges[idx] for idx in probe_atoms]
    query_esp_charges = [query.charges[idx] for idx in query_atoms]

    probe_self_integral = calculate_gaussian_integrals(distance_probe_probe, probe_esp_charges, probe_esp_charges)
    query_self_integral = calculate_gaussian_integrals(distance_query_query, query_esp_charges, query_esp_charges)
    probe_query_integral = calculate_gaussian_integrals(distance_probe_query, probe_esp_charges, query_esp_charges)

    return calculate_similarity(
        probe_self_integral, query_self_integral, probe_query_integral, metric
    )
