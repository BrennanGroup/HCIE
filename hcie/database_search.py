import os
import json
from pathlib import Path
from datetime import datetime
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

import hcie
from hcie import Molecule, Alignment

# Define paths to package data
package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = Path(package_dir) / "Data"
smiles_json_path = Path(data_dir_path) / "vehicle_smiles.json"

# Load required data
with open(smiles_json_path, "r") as json_file:
    vehicle_dict = json.load(json_file)


def vehicle_search_parallel(
    query_smiles: str,
    query_mol_vector: (int, int) = None,
    query_name: str = None,
    num_of_output_mols: int = 50,
):
    """
    Search a query molecule against the molecules in the VEHICLe database, aligning to the MedChem vector designated by
    query_mol_vector_id. To determine the appropriate vector it is necessary to run query_mol.write_vectors_to_image()
    and examine the image produced.


    Parameters
    ----------
    query_name: Name of molecule, optional.
    query_smiles: SMILES string of molecule to search against the VEHICLe database.
    query_mol_vector: Functionalisable bond to align to (non-H atom ID, H-atom ID)
    num_of_output_mols: number of mols to output to sdf file. Defaults to 20

    Returns
    -------
    results:
    aligned_mols
    """
    query_mol = Molecule(query_smiles, name=query_name)
    if query_mol_vector is None:
        if query_mol.alignment_vector is not None:
            query_mol_vector = query_mol.alignment_vector
        else:
            raise ValueError("Query Vector must be supplied for alignment")

    aligned_mols = multiprocessing.Manager().dict()

    aligned_mols["query_mol"] = query_mol

    results = database_search_parallel(
        query_mol,
        query_mol_vector=query_mol_vector,
        probe_dict=vehicle_dict,
        database_mols=aligned_mols,
    )

    query_mol.replace_hydrogen_with_dummy_atom(
        query_mol.alignment_vector[1], query_mol.rdmol
    )
    aligned_mols["query_with_dummy"] = query_mol
    results_to_sdf(results, aligned_mols, num_of_mols=num_of_output_mols)
    print_results(results, aligned_mols, query_smiles=query_smiles, query_name=f"{query_name}")

    return None


def align_and_score_probe(
    probe_regid: str,
    probe_smiles: str,
    query_mol: hcie.Molecule,
    query_mol_vector: (int, int),
    similarity_metric: str = "tanimoto",
):
    """
    Aligns a probe molecule onto the query molecule along each of the functionalisable vectors in the probe molecule,
    scores each, and saves each alignment with their scores. Returns the highest score.
    Parameters
    ----------
    probe_regid: VEHICLe regid of probe molecule.
    probe_smiles: SMILES string of probe molecule.
    query_mol: HCIE.molecule of query molecule.
    query_mol_vector: Vector in form (non-H atom idx, H atom idx) of query vector to align to.
    similarity_metric: Metric to use for computing ESP similarity - defaults to Tanimoto.

    Returns
    -------
    (probe_regid, best_score, best_idx, best_esp, best_shape, probe_mol: hcie.Molecule)
    """
    probe = Molecule(probe_smiles, name=str(probe_regid))

    probe_vectors = probe.functionalisable_bonds

    best_score = best_esp = best_shape = best_vector = 0
    best_idx = None

    for conf_idx, vector in enumerate(probe_vectors):
        # Need to double the conf_idx because each alignment generates two conformers - the aligned one, and its 180-
        # degree rotation
        alignment = Alignment(
            probe_molecule=probe,
            reference_molecule=query_mol,
            reference_bond_idxs=query_mol_vector,
            probe_bond_idxs=vector,
            probe_conf_id=2 * conf_idx,
        )
        _, _ = alignment.align_score(similarity_metric=similarity_metric)

        for idx in (2 * conf_idx, 2 * conf_idx + 1):

            total_score = probe.total_scores[idx]

            if total_score > best_score:
                best_score = total_score
                best_esp = probe.esp_scores[idx]
                best_shape = probe.shape_scores[idx]
                best_idx = idx
                best_vector = vector

    if best_idx is not None:
        probe.replace_hydrogen_with_dummy_atom(atom_id=best_vector[1], mol=probe.rdmol)
        return probe_regid, best_score, best_idx, best_esp, best_shape, probe


def database_search_parallel(
    query_mol: hcie.Molecule,
    query_mol_vector: (int, int),
    probe_dict: dict,
    database_mols: dict,
    similarity_metric: str = "tanimoto",
):
    """

    Parameters
    ----------
    query_mol: The molecule to use as a reference molecule for the search
    query_mol_vector: Functionalisable bond to align to (non-H atom ID, H-atom ID)
    probe_dict: dictionary of probe molecule identifiers (keys) and SMILES strings (values)
    database_mols: a blank dictionary to which probe molecule identifiers (keys) and hcie molecules (values) will be
                    appended.
    similarity_metric: Similarity metric to use when scoring ESP similarity - defaults to Tanimoto

    Returns
    -------
    Results: a list of tuples of results in the form (probe_id, best_score, best_idx, best_esp, best_shape) in order of
    best combined score to worst.
    """

    results_list = []

    # Use ThreadPoolExecutor to parallelise processing of VEHICLe.
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                align_and_score_probe, entry[0], entry[1], query_mol, query_mol_vector
            )
            for entry in probe_dict.items()
        ]

        for future in tqdm(as_completed(futures), total=24867, desc="Searching"):
            result = future.result()
            if result:
                results_list.append(result[:-1])
                database_mols[result[0]] = result[-1]

    return sorted(results_list, key=lambda x: x[1], reverse=True)


def new_directory(new_dir):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Save the current working directory
            current_dir = os.getcwd()

            try:
                # Create the new directory if it doesn't exist
                os.makedirs(new_dir, exist_ok=True)

                # Change the working directory to the new directory
                os.chdir(new_dir)

                # Call the wrapped function
                result = func(*args, **kwargs)
                return result
            finally:
                # Change back to the previous directory after the function finishes
                os.chdir(current_dir)

        return wrapper

    return decorator


@new_directory("hcie_results")
def print_results(results_list: list, database_mols: dict, query_smiles: str, query_name: str):
    """
    Prints out the results of a search against the VEHICLe database.
    Parameters
    ----------
    results_list: list of tuples (REGId (key), [combined score, confId of best match, ESP Score, Shape Score]).
    database_mols: dictionary of probe molecules in the alignment of best similarity to the probe.
    query_smiles: The SMILES string of the query molecule
    query_name: The name of the query molecule.

    Returns
    -------

    """
    current_datetime = datetime.now()
    with open(f"{query_name}_hcie.txt", "w") as output_file:
        # Write the title line
        output_file.write(
            f'Generated by HCIE at {current_datetime.strftime("%H:%M, %d/%m/%Y")}'
            + "\n"
        )

        # Write the query file out
        output_file.write(f"Query molecule: {query_name}" + "\n")
        output_file.write(f"Query SMILES: {query_smiles}" + "\n")

        output_headers = [
            "RegID",
            "SMILES",
            "Score",
            "ESP Score",
            "Shape Score",
            "Conformer ID",
        ]

        output_file.write(
            "-" * 82
            + "\n"
            + f"{output_headers[0]:6}\t{output_headers[1]:30}\t{output_headers[2]:3}\t{output_headers[3]:3}\t{output_headers[4]:3}\t{output_headers[5]}"
            + "\n"
            + "-" * 82
            + "\n"
        )
        for item in results_list:
            regid, score, conf_id, esp_score, shape_score = (
                item[0],
                float(item[1]),
                item[2],
                float(item[3]),
                float(item[4]),
            )

            output_smiles = Chem.MolToSmiles(Chem.RemoveHs(database_mols[regid].rdmol))
            row_line = f"{regid:6}\t{output_smiles:30}\t{score:<5.2f}\t{esp_score:<9.2f}\t{shape_score:<11.2f}\t{conf_id}"
            output_file.write(row_line + "\n")

    return None


@new_directory(
    "../../HCIE_v2_testing/ChEMBL_analysis/Heterocycles Feb 2024/hcie_results"
)
def print_xyz_files(
    results_list: list[tuple],
    reference: hcie.Molecule,
    mol_dict: dict,
    num_of_mols: int = 20,
):
    """
    Prints the XYZ files of the top num_of_mols molecules in their best alignment with the query molecule
    Parameters
    ----------
    results_list: list of tuples (REGId (key), [combined score, confId of best match, ESP Score, Shape Score]) in order
    best to worst match
    reference: HCIE.Molecule - reference molecule
    mol_dict: dictionary of REGIds (keys) and HCIE.molecule objects for each molecule
    num_of_mols: How many of the top molecules to print to XYZ files - default is 20

    Returns
    -------
    None
    """
    print_results(results_list=results_list, query=reference)

    Chem.MolToXYZFile(reference.rdmol, filename="query.xyz", confId=0)

    for item in results_list[:num_of_mols]:
        regid, _, conf_id, _, _ = (
            item[0],
            float(item[1]),
            item[2],
            float(item[3]),
            float(item[4]),
        )

        Chem.MolToXYZFile(
            mol_dict[regid].rdmol, filename=f"{regid}.xyz", confId=conf_id
        )

    return None


@new_directory("hcie_results")
def results_to_sdf(results_list: list, aligned_mols: dict, num_of_mols: int):
    """
    Prints the results of a HCIE search to a txt file and the top molecules (as defined by num_of_mols) to sdf file.
    Parameters
    ----------
    results_list: Results list returned by HCIE search
    aligned_mols: dictionary of mols aligned to query mol by HCIE
    num_of_mols: Number of top molecules to include in SDF file.

    Returns
    -------

    """
    filename = f'{aligned_mols["query_mol"].name}_results.sdf'

    with Chem.SDWriter(filename) as writer:
        # Write the query mol
        query = aligned_mols["query_with_dummy"]
        query.rdmol.SetProp("_Name", "Query")
        writer.write(query.rdmol)

        # Write the aligned VEHICLe mols to the SDF
        for result in results_list[:num_of_mols]:
            mol = aligned_mols[result[0]]
            mol.rdmol.SetProp("_Name", f"{result[0]}")
            mol.rdmol.SetProp("Total_score", f"{result[1]}")
            mol.rdmol.SetProp("ESP_similarity", f"{result[3]}")
            mol.rdmol.SetProp("Shape_similarity", f"{result[4]}")
            writer.write(mol.rdmol, confId=int(result[2]))

    return None
