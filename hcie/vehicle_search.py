import time
import json
import multiprocessing
import importlib.resources
from typing import Optional, Union, List, Tuple
import random

from hcie.molecule import Molecule
from hcie.alignment import AlignmentOneVector, AlignmentTwoVector
from hcie.outputs import print_results, alignments_to_sdf, mols_to_image


# Load full database into memory
def load_database():
    with importlib.resources.files("Data").joinpath(
        "MoBiVic_2.json"
    ).open("r") as json_file:
        data = json.load(json_file)
    return data


# Load dictionary looking up molecules by hash
with importlib.resources.files("Data").joinpath("mobivic_by_hash.json").open(
    "r"
) as json_file:
    vehicle_by_hash = json.load(json_file)


class VehicleSearch:
    def __init__(
        self,
        smiles: Optional[str] = None,
        name: str = None,
        charges: list | None = None,
        query_vector: list | tuple | None = None,
        shape_weighting: float = 0.5,
        esp_weighting: float = 0.5,
        xyz_block: Optional[str] = None,
    ):
        if not smiles and not xyz_block:
            raise ValueError("Either SMILES or an XYZ block must be provided")

        self.smiles = smiles
        self.name = name
        self.query = Molecule(
            smiles=smiles,
            name=name,
            charges=charges,
            query_vector=query_vector,
            xyz_block=xyz_block,
        )

        self.query_hash = (
            self.query.user_vector_hash
            if self.query.user_vector_hash is not None
            else None
        )

        self.charge_type = "Gasteiger" #if charges is None else "orca_charges"
        self.query_charges = charges
        self.shape_weight = shape_weighting
        self.esp_weight = esp_weighting

        if self.query_hash is not None:
            self.hash_matches = self.search_vehicle_by_hash()
            self.vehicle_vector_matches = {}
            self.search_type = "hash"
        elif self.query_hash is None and len(self.query.user_vectors) > 0:
            self.search_type = "vector"
        else:
            raise ValueError("No user exit-vectors available for search")

    def _query_label(self):
        if self.query.smiles:
            return self.query.smiles
        return f"XYZ:{self.query.name or 'query'}"

    def search(self, database_size: int | None = None):
        """
        Top-level function in the class - coordinates database searching functionality, and handles overall
        multiprocessing management.

        Follows this logic:
            1. Identify whether a single vector ('vector') search or a two-vector ('hash') search is being requested.
            2. Carry out the required search, and collect the results
            3. Print the results and the alignments to file, ordered from highest scoring to lowest scoring.
            4. Print the requested number of alignments to SDF file.
        :return: None
        """
        print(f"Searching for {self._query_label()}")
        start = time.time()
        with multiprocessing.Manager() as manager:
            full_db = load_database()

            # Optionally subsample for runtime benchmarking
            if database_size is not None and database_size < len(full_db):
                selected_regids = random.sample(list(full_db.keys()), database_size)
                db_subset = {regid: full_db[regid] for regid in selected_regids}
                database_by_regid = manager.dict(db_subset)
            else:
                database_by_regid = manager.dict(full_db)

            # One-vector search
            if self.search_type == "vector":
                results, mols = self.align_and_score_vector_matches(database_by_regid)

            # Two-vector search
            elif self.search_type == "hash":
                self.vehicle_vector_matches = self.get_exit_vectors_for_hash_matches(
                    database_by_regid
                )
                results, mols = self.align_and_score_hash_matches_pooled(
                    database_by_regid
                )

            else:
                raise ValueError("Search type not supported")

        self.results_to_file(results, mols)

        finish = time.time()
        print(f"Search completed in {round(finish - start, 2)} seconds")

        return finish - start

    def results_to_file(self, results: list, mols: dict) -> None:
        """
        Prints the results of the search to a txt/csv file, the alignments to an sdf file, and generates a png image
        of the top 50 mols returned for ease of viewing.
        Parameters
        ----------
        results: list of the results, sorted from most similar to least similar
        mols: dictionary of the molecule objects of the top aligned mols

        Returns
        -------
        None
        """
        mols["query"] = self.query
        query_label = self.query.smiles if self.query.smiles else f"<XYZ:{self.query.name or 'query'}>"
        print_results(
            mols, results, query_smiles=query_label, query_name=self.query.name
        )
        alignments_to_sdf(
            results=results, mol_alignments=mols, query_name=self.query.name
        )
        mols_to_image(results, query_name=self.query.name, num_of_mols=50)

        return None

    def align_and_score_vector_matches(
        self, database_by_regid: dict
    ) -> tuple[list, dict]:
        """
        One-vector alignment logic.

        Alignment method when only one exit vector is specified by the user.
        Method is as follows:
            1. For each probe molecule, identify the exit-vectors in the molecule
            2. Align each exit-vector in the probe to the user-specified query exit-vector.
            3. Score this alignment
            4. Rotate the alignment by 180 degrees about the exit-vector and then rescore.
            5. Repeat this for every exit-vector on the probe molecule, and then return the highest scoring alignment.
        :return:
        List of results, sorted by highest total score, and dictionary of Molecule objects for the processed molecules.
        """
        print("Aligning to all database molecules")

        with multiprocessing.Pool() as pool:
            task_args = self.generate_single_vector_tasks(database_by_regid)

            results = list(
                pool.imap_unordered(
                    self.align_and_score_probe_by_vector_wrapper,
                    task_args,
                    chunksize=15000,
                )
            )

        processed_mols = {result[0]: result[-1] for result in results}
        results = [result[:-1] for result in results]

        return sorted(results, key=lambda x: x[1], reverse=True), processed_mols

    def generate_single_vector_tasks(self, database_by_regid: dict) -> list:
        """
        Generates the task arguments needed to parallelise the single-vector aligning and scoring of the database
        molecules to the query ligand
        Parameters
        ----------
        database_by_regid: dictionary of molecular properties, keyed by RegID

        Returns
        -------
        List of task arguments ((RegID, SMILES), {"similarity_metric": sim_metric})
        """
        # Default behaviour should be using Gasteiger charges
        if self.charge_type == "Gasteiger":
            task_args = [
                ((regid, properties["smiles"]), {"similarity_metric": "Tanimoto"})
                for regid, properties in database_by_regid.items()
            ]
        # If other, user-specified charges are used, ensure that these are within the database.
        else:
            try:
                task_args = [
                    (
                        (regid, properties["smiles"], properties[self.charge_type]),
                        {"similarity_metric": "Tanimoto"},
                    )
                    for regid, properties in database_by_regid.items()
                ]
            except KeyError:
                raise ValueError(
                    f"charge_type {self.charge_type} is not in the database"
                )

        return task_args

    def align_and_score_probe_by_vector_wrapper(self, all_args):
        """
        Wrapper for one-vector alignment logic.
        """
        args, kwargs = all_args
        return self.align_and_score_probe_by_vector(*args, **kwargs)

    def align_and_score_probe_by_vector(
        self,
        probe_regid: str,
        probe_smiles: str,
        charges: list[float] | None = None,
        similarity_metric: str = "Tanimoto",
    ) -> list:
        """
        One-vector alignment logic.

        Method to align a probe molecule specified by RegID and SMILES string to a query molecule with a single,
        user-specified exit-vector.

        Logic:
                1. Generates twice as many conformers in the probe molecule as there are probe exit-vectors
                2. Loops through each exit vector, aligning the exit vector to the query exit-vector,
                and then aligning the ring planes. This alignment is scored and the score stored, alongside the
                coordinates of the alignment.
                3. The alignment is then flipped about the axis of the exit-vector by 180 degrees, and re-scored.
                This alignment is then stored alongside the score.
                4. The highest total-scoring alignment is then returned, alongside the scores.

        :param probe_regid: the RegID of the probe to align and score against the query
        :param probe_smiles: The SMILES string of the probe molecule
        :param charges: list of charges (optional) - this is for if charges other than Gasteiger charges are required by
        the user
        :param similarity_metric: similarity metric to use in the scoring - Tanimoto is highly recommended here.
        :return: List of scores and information about the probe
                    [
                    RegID,
                    highest total score,
                    shape score of highest overall scoring alignment
                    ESP score of highest overall scoring alignment
                    Conformer index of highest overall scoring alignment
                    SMILES string of highest overall scoring alignment, with attachment points indicated with dummies
                    hcie.Molecule of probe molecule
                    ]
        """
        probe = Molecule(probe_smiles, charges=charges)
        probe.generate_conformers(2 * len(probe.exit_vectors))

        for conf_idx, vector in enumerate(probe.exit_vectors):
            alignment = AlignmentOneVector(
                probe_molecule=probe,
                query_molecule=self.query,
                query_exit_vectors=self.query.user_vectors[0],
                probe_exit_vectors=vector,
                probe_conformer_idx=2 * conf_idx,
            )
            alignment.align_and_score(similarity_metric=similarity_metric)

        best_conf_idx = max(probe.total_scores, key=probe.total_scores.get)
        # Need to floor divide because each vector alignment has two possible orientations, but these both correspond
        # to the same exit vector
        best_vector = probe.exit_vectors[best_conf_idx // 2]

        # Convert to SMILES
        best_smiles = self._vectors_to_dummies(probe, best_vector, update_mol=True)

        return [
            probe_regid,
            probe.total_scores[best_conf_idx],
            probe.shape_scores[best_conf_idx],
            probe.esp_scores[best_conf_idx],
            best_conf_idx,
            best_smiles,
            probe,
        ]

    def search_vehicle_by_hash(self) -> list:
        """
        Two-vector alignment logic.

        Search the vehicle database by hash
        :return: list of regids matched by hash
        """
        return vehicle_by_hash[self.query_hash]

    def get_exit_vectors_for_hash_matches(self, database_by_regid: dict) -> dict:
        """
        Two-vector alignment logic.

        For each of the VEHICLe results found by matching the query hash to those of the VEHICLe hashes, get the atom
        IDs of the bonds that correspond to the hash match.

        For example:
            Query Hash = 00111011
            A search against VEHICLe reveals 4268 matches, one of which is S230.
            This function will return the atom IDs in S230 that correspond to the hash
                    00111011 =  ((0,8), (3,10))
                                ((1,9), (4, 11))
                                ((3, 10), (7, 12))

            These can then be used to align and score the query to the ligand.
        :return: Dictionary keyed by regid, and values as a list of the exit vectors in the regid corresponding to
        the query hash.
        """
        return {
            match: [
                vector["vectors"]
                for vector in database_by_regid[match]["exit_vectors"][self.query_hash]
            ]
            for match in self.hash_matches
        }

    def align_and_score_vehicle_molecule(
        self, regid: str, vector_pairs: list, database_by_regid: dict
    ) -> list:
        """
        Two-vector alignment logic.

        Align and score the vector pairs that match the hash searched for a single regid.
        :param regid: RegID of probe (molecule to align against query)
        :param vector_pairs: List of vector pairs in the probe molecule that match the query hash
        :param database_by_regid: Dictionary of database molecules, keyed by RegID
        :return: List of scores and information about the probe
                    [
                    RegID,
                    highest total score,
                    shape score of highest overall scoring alignment
                    ESP score of highest overall scoring alignment
                    Conformer index of highest overall scoring alignment
                    SMILES string of highest overall scoring alignment, with attachment points indicated with dummies
                    hcie.Molecule of probe molecule
                    ]
        """
        probe = self.initialise_probe_molecule(
            regid, len(vector_pairs), database_by_regid
        )

        # Loop through each vector pair
        for idx, vector_pair in enumerate(vector_pairs):
            # Align and score both orientations of the vector pair
            self.align_and_score_orientation(
                probe=probe, probe_vector_pair=vector_pair, conformer_idx=2 * idx
            )

            # Now flip the vector and align again
            flipped_vector = [vector_pair[1], vector_pair[0]]
            self.align_and_score_orientation(
                probe=probe, probe_vector_pair=flipped_vector, conformer_idx=2 * idx + 1
            )
        # Determine the best conformer index and therefore the highest matching vector pair.
        best_conf_idx = max(probe.total_scores, key=probe.total_scores.get)
        best_vector = self._get_best_vector(vector_pairs, best_conf_idx)
        best_smiles = self._vectors_to_dummies(probe, best_vector, update_mol=True)

        return [
            regid,
            probe.total_scores[best_conf_idx],
            probe.shape_scores[best_conf_idx],
            probe.esp_scores[best_conf_idx],
            best_conf_idx,
            best_smiles,
            probe,
        ]

    def align_and_score_molecule_wrapper(self, args):
        return self.align_and_score_vehicle_molecule(*args)

    def align_and_score_hash_matches_pooled(
        self, database_by_regid: dict
    ) -> tuple[list, dict]:
        """
        Method for handling the two-vector alignment to hash matches.

        All molecules with a two-exit-vector geometry that matches the query hash are aligned to the query along
        these vectors and scored accordingly.

        Molecules are distributed across processes in parallel using multiprocessing.Pool class for maximum efficiency.
        Parameters
        ----------
        database_by_regid: Dictionary of database molecules, keyed by RegID - this needs to be passed to each process to
        avoid overloading the memory by loading it in each time.

        Returns
        -------
        List of results, sorted by highest total score, and dictionary of Molecule objects for the processed molecules.
        """

        print(f"Aligning to {len(self.vehicle_vector_matches)} vector matches")

        with multiprocessing.Pool() as pool:
            task_args = [
                (match_regid, vector_pairs, database_by_regid)
                for match_regid, vector_pairs in self.vehicle_vector_matches.items()
            ]
            results = list(
                pool.imap_unordered(
                    self.align_and_score_molecule_wrapper,
                    task_args,
                    chunksize=len(self.vehicle_vector_matches) // 8 + 1,
                )
            )

        processed_mols = {result[0]: result[-1] for result in results}
        results = [result[:-1] for result in results]

        return sorted(results, key=lambda x: x[1], reverse=True), processed_mols

    def _vectors_to_dummies(
        self, probe: Molecule, vector_pair: list | tuple, update_mol: bool = False
    ) -> str:
        """
        Take a probe molecule and the vector pair of highest scoring alignment, and return a SMILES string with the
        attachment points indicated with dummy atoms
        :param probe: probe molecule
        :param vector_pair: list of lists of atom indices of vectors of highest alignment
        :return: SMILES string of molecule
        """
        if self.search_type == "hash":
            hydrogens_to_replace = [vector[1] for vector in vector_pair]

        elif self.search_type == "vector":
            hydrogens_to_replace = [vector_pair[1]]

        else:
            raise ValueError(f"Search type {self.search_type} not supported")

        return probe.replace_hydrogens_with_dummy_atoms(
            hydrogens_to_replace, update_mol=update_mol
        )

    @staticmethod
    def _get_best_vector(vector_pairs: list, best_conf_idx: int) -> list:
        """
        Private method to return the best vector pair from a conformer index.

        The alignments are generated according to the order of the vectors in the user_vector list, for example:
            vectors matching hash in probe (A, B)
            A is aligned to query vector 1, B to query vector 2, and scored. These scores are stored in conformer
            with idx x (where x is even)
            B is then aligned to query vector 1, A to query vector 2, and scored. These scores are stored in
            conformer with idx x + 1 (therefore an odd numbered conformer).

        Therefore, if the conformer index with the highest scoring alignment is even, the vectors must be returned in
        the order that they appear originally in the user_vector list (in the example above (A, B)).

        If the confomer idx is odd, then the order must be flipped relative to the order in the original user_vector
        list (in the above examble (B, A) would need to be returned).

        :param vector_pairs: List of vector pairs (A, B) with the highest scoring alignment
        :return: list of vector pairs, in the order that has generated the higest scoring alignment.
        """
        if best_conf_idx % 2 == 0:
            return vector_pairs[best_conf_idx // 2]
        else:
            return [
                vector_pairs[best_conf_idx // 2][1],
                vector_pairs[best_conf_idx // 2][0],
            ]

    def align_and_score_orientation(
        self, probe: Molecule, probe_vector_pair, conformer_idx
    ) -> None:
        """
        Two-vector (Hash) alignment logic.

        Aligns a pair of vectors in a probe molecule to the user-specified pair of vectors in the query molecule,
        and scores the alignment, storing these scores in dictionaries (implemented as attributes of the probe
        molecule) keyed by conformer idx.

        :param probe: probe molecule to align to query
        :param probe_vector_pair: probe vector pair to align to query vector pair
        :param conformer_idx: probe conformer to store alignment results - this serves as an RDKit conformer index to
        store the coordinates of the alignment, and the key in the scoring dictionaries where the score is stored.
        :return: None
        """
        alignment = AlignmentTwoVector(
            query_molecule=self.query,
            probe_molecule=probe,
            query_exit_vectors=self.query.user_vectors,
            probe_exit_vectors=probe_vector_pair,
            probe_conformer_idx=conformer_idx,
            shape_weighting=self.shape_weight,
            esp_weighting=self.esp_weight,
        )

        # Retrieve and store scores

        shape_sim, esp_sim = alignment.align_and_score()
        probe.shape_scores[conformer_idx] = shape_sim
        probe.esp_scores[conformer_idx] = esp_sim
        probe.total_scores[conformer_idx] = alignment.calculate_total_score(
            shape_sim, esp_sim
        )

        return None

    def initialise_probe_molecule(
        self, regid: str, num_of_vector_pairs: int, database_by_regid: dict
    ) -> Molecule:
        """
        Two-vector (hash) alignment logic.

        Initialise probe molecule specified by RegID with the appropriate number of conformers (twice the number of
        vector pairs matching the hash of the user-specified exit-vectors on the query molecule)

        :param regid: the regid of the probe molecule
        :param num_of_vector_pairs: The number of vector pairs matching the searched hash of the user-specified
        exit-vectors
        :param database_by_regid: dictionary of database molecule info (SMILES strings etc.) keyed by RegID.
        :return: instantiated Molecule
        """
        probe = Molecule(
            database_by_regid[regid]["smiles"],
            charges=(
                None
                if self.charge_type == "Gasteiger"
                else database_by_regid[regid][self.charge_type]
            ),
        )
        probe.generate_conformers(num_confs=2 * num_of_vector_pairs)

        return probe
