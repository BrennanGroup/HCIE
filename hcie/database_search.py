import os
import json
from pathlib import Path
from datetime import datetime
from rdkit import Chem

import hcie
from hcie import Molecule, Alignment

# Define paths to package data
package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = Path(package_dir) / "Data"
smiles_json_path = Path(data_dir_path) / "vehicle_smiles.json"

# Load required data
with open(smiles_json_path, 'r') as json_file:
    vehicle_dict = json.load(json_file)


def vehicle_search(query_smiles: str,
                   query_mol_vector: (int, int),
                   query_name: str = None,
                   num_of_output_mols: int = 20
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
    None
    """
    query_mol = Molecule(query_smiles, name=query_name)
    aligned_mols = {}

    results = database_search(query_mol,
                              query_mol_vector=query_mol_vector,
                              probe_dict=vehicle_dict,
                              database_mols=aligned_mols)

    # write_to_sdf_file(results_list=results,
    #                   reference=query_mol,
    #                   mol_dict=aligned_mols,
    #                   num_of_mols=num_of_output_mols)

    return None


def database_search(query_mol: hcie.Molecule,
                    query_mol_vector: (int, int),
                    probe_dict: dict,
                    database_mols: dict,
                    similarity_metric: str = 'tanimoto'
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

    reference_vector = query_mol_vector

    for entry in probe_dict.items():
        probe_regid, probe_smiles = entry

        probe = Molecule(probe_smiles, name=f'{probe_regid}')
        database_mols[probe_regid] = probe

        probe_vectors = probe.functionalisable_bonds

        best_score = best_esp = best_shape = 0
        best_idx = None

        print(probe_regid)
        for conf_idx, vector in enumerate(probe_vectors):
            alignment = Alignment(probe_molecule=probe,
                                  reference_molecule=query_mol,
                                  reference_bond_idxs=reference_vector,
                                  probe_bond_idxs=vector,
                                  probe_conf_id=conf_idx)

            esp_score, shape_score = alignment.align_score(similarity_metric=similarity_metric)
            total_score = esp_score + shape_score

            if total_score > best_score:
                best_score = total_score
                best_esp = esp_score
                best_shape = shape_score
                best_idx = conf_idx

        if best_idx is not None:
            results_list.append(
                (probe_regid, best_score, best_idx, best_esp, best_shape)
            )

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


def print_results(results_list, query):
    """
    Prints out the results of a search against the VEHICLe database.
    Parameters
    ----------
    results_list: list of tuples (REGId (key), [combined score, confId of best match, ESP Score, Shape Score]).
    query: The SMILES string of the query molecule

    Returns
    -------

    """
    current_datetime = datetime.now()
    with open(f"{query}_hcie.txt", "w") as output_file:
        # Write the title line
        output_file.write(
            f'Generated by HCIE at {current_datetime.strftime("%H:%M, %d/%m/%Y")}'
            + "\n"
        )

        # Write the query file out
        output_file.write(f"Query molecule: {query}" + "\n")

        output_file.write(
            "-" * 50
            + "\n"
            + "RegID\tScore\tESP Score\tShape Score\tConformer ID"
            + "\n"
            + "-" * 50
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
            row_line = f"{regid:6}\t{score:3.2f}\t{esp_score:3.2f}\t{shape_score:3.2f}\t{conf_id}"
            output_file.write(row_line + "\n")

    return None


@new_directory("hcie_results")
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
def write_to_sdf_file(
    results_list: list[tuple],
    reference: hcie.Molecule,
    mol_dict: dict,
    num_of_mols: int = 20,
):
    """
    Prints the SDF files of the top num_of_mols molecules in their best alignment with the query molecule
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

    writer = Chem.SDWriter("hcie_results.sdf")

    writer.write(reference.rdmol, confId=0)

    for item in results_list[:num_of_mols]:
        regid, _, conf_id, _, _ = (
            item[0],
            float(item[1]),
            item[2],
            float(item[3]),
            float(item[4]),
        )

        writer.write(mol_dict[regid].rdmol, confId=conf_id)

    return None


if __name__ == "__main__":
    start_time = datetime.now()
    vehicle_search(query_smiles='c2ccc1ncccc1c2',
                   query_mol_vector=(2, 12),
                   query_name='quinoline'
                   )
    end_time = datetime.now()
    print(f'{end_time - start_time} seconds')
