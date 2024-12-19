"""
Script to search the VEHICLe database by Hash

Written by Matthew Holland on 4 September 2024
"""

import time
import json
from tqdm import tqdm
import multiprocessing
import importlib.resources

from hcie.molecule import Molecule
from hcie.alignment import AlignmentOneVector, AlignmentTwoVector
from hcie.outputs import print_results, alignments_to_sdf, mols_to_image


def load_database():
    with importlib.resources.files('Data').joinpath('bifunctionalised_database_no_peb.json').open('r') as json_file:
        data = json.load(json_file)
    return data

with importlib.resources.files('Data').joinpath('database_by_hash.json').open('r') as json_file:
    vehicle_by_hash = json.load(json_file)


class VehicleSearch:
    def __init__(self,
                 smiles: str,
                 name: str = None,
                 charges: list | None = None,
                 query_vector: list | tuple | None = None,
                 shape_weighting: float = 0.5,
                 esp_weighting: float = 0.5
                 ):

        self.smiles = smiles
        self.name = name
        self.query = Molecule(smiles, name=name, charges=charges, query_vector=query_vector)
        self.query_hash = self.query.user_vector_hash if self.query.user_vector_hash is not None else None
        self.charge_type = "Gasteiger" if charges is None else 'orca_charges'
        self.query_charges = charges
        self.shape_weight = shape_weighting
        self.esp_weight = esp_weighting

        if self.query_hash is not None:
            self.hash_matches = self.search_vehicle_by_hash()
            self.vehicle_vector_matches = {}
            self.search_type = "hash"
        elif self.query_hash is None and len(self.query.user_vectors) > 0 :
            self.search_type = "vector"

    def search_vehicle_by_hash(self) -> list:
        """
        Search the vehicle database by hash
        :return: list of regids matched by hash
        """
        return vehicle_by_hash[self.query_hash]

    def get_exit_vectors_for_hash_matches(self, database_by_regid: dict)->dict:
        """
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
        :return:
        """
        return {
            match: [vector['vectors'] for vector in database_by_regid[match]['exit_vectors'][self.query_hash]]
            for match in self.hash_matches
        }

    def align_and_score_vehicle_molecule(self, regid: str, vector_pairs: list, database_by_regid: dict) -> list:
        """
        Align and score the vector pairs that match the hash searched for a single regid.
        :param regid:
        :param vector_pairs:
        :return:
        """
        probe = self.initialise_probe_molecule(regid, len(vector_pairs), database_by_regid)

        # Loop through each vector pair
        for idx, vector_pair in enumerate(vector_pairs):
            # Align and score both orientations of the vector pair
            self.align_and_score_orientation(probe=probe,
                                             probe_vector_pair=vector_pair,
                                             conformer_idx=2*idx)

            # Now flip the vector and align again
            flipped_vector = [vector_pair[1], vector_pair[0]]
            self.align_and_score_orientation(probe=probe,
                                             probe_vector_pair=flipped_vector,
                                             conformer_idx=2*idx+1)
        # Determine the best conformer index and therefore the highest matching vector pair.
        best_conf_idx = max(probe.total_scores, key=probe.total_scores.get)
        best_vector = self._get_best_vector(vector_pairs, best_conf_idx)
        best_smiles = self._vectors_to_dummies(probe, best_vector, update_mol=True)

        return [regid, probe.total_scores[best_conf_idx], probe.shape_scores[best_conf_idx],
                probe.esp_scores[best_conf_idx], best_conf_idx, best_smiles, probe]

    def search(self):
        """

        :return:
        """
        print(f'Searching for {self.smiles}')
        start = time.time()
        with multiprocessing.Manager() as manager:
            database_by_regid = manager.dict(load_database())

            if self.search_type == "hash":
                self.vehicle_vector_matches = self.get_exit_vectors_for_hash_matches(database_by_regid)
                results, mols = self.align_and_score_pooled(database_by_regid)
            elif self.search_type == "vector":
                results, mols = self.align_and_score_vector_matches(database_by_regid)
            else:
                raise ValueError("Search type not supported")

        mols["query"] = self.query
        print_results(results,
                      query_smiles=self.query.smiles,
                      query_name=self.query.name)
        alignments_to_sdf(results=results,
                          mol_alignments=mols,
                          query_name=self.query.name)
        mols_to_image(results,
                      query_name=self.query.name,
                      num_of_mols=50)
        finish = time.time()
        print(f'Search completed in {round(finish - start, 2)} seconds')
        return None

    def align_and_score_molecule_wrapper(self, args):
        return self.align_and_score_vehicle_molecule(*args)

    def align_and_score_pooled(self, database_by_regid: dict) -> tuple[list, dict]:

        print(f'Aligning to {len(self.vehicle_vector_matches)} vector matches')

        with multiprocessing.Pool() as pool:
            task_args = [
               (match_regid, vector_pairs, database_by_regid)
                for match_regid, vector_pairs in self.vehicle_vector_matches.items()
           ]
            results = list(tqdm(
                pool.imap_unordered(self.align_and_score_molecule_wrapper, task_args, chunksize=len(
                    self.vehicle_vector_matches)//8 +1),
                total=len(task_args),
                desc="Searching"
            ))

        processed_mols = {result[0]: result[-1] for result in results}
        results = [result[:-1] for result in results]

        return sorted(results, key=lambda x: x[1], reverse=True), processed_mols

    def align_and_score_vector_matches(self, database_by_regid: dict) -> tuple[list, dict]:
        """
        Alignment method when only one exit vector is specified by the user.
        Method is as follows:
            1. For each probe molecule, identify the exit-vectors in the molecule
            2. Align each exit-vector in the probe to the user-specified query exit-vector.
            3. Score this alignment
            4. Rotate the alignment by 180 degrees about the exit-vector and then rescore.
            5. Repeat this for every exit-vector on the probe molecule, and then return the highest scoring alignment.
        :return:
        """
        print('Aligning to all database molecules')

        with multiprocessing.Pool() as pool:
            if self.charge_type == 'Gasteiger':
                task_args = [
                    ((regid, properties['smiles']),{"similarity_metric": "Tanimoto"})
                    for regid, properties in database_by_regid.items()
                ]
            else:
                try:
                    task_args = [
                        ((regid, properties['smiles'], properties[self.charge_type]),{'similarity_metric': 'Tanimoto'})
                        for regid, properties in database_by_regid.items()
                    ]
                except KeyError:
                    raise ValueError(f"charge_type {self.charge_type} is not in the database")

            results = list(
                pool.imap_unordered(self.align_and_score_probe_by_vector_wrapper,
                                    task_args,
                                    chunksize=15000)
            )

        processed_mols = {result[0]: result[-1] for result in results}
        results = [result[:-1] for result in results]

        return sorted(results, key=lambda x: x[1], reverse=True), processed_mols

    def align_and_score_probe_by_vector_wrapper(self, all_args):
        args, kwargs = all_args
        return self.align_and_score_probe_by_vector(*args, **kwargs)

    def align_and_score_probe_by_vector(self,
                                        probe_regid: str,
                                        probe_smiles: str,
                                        charges: list[float] | None = None,
                                        similarity_metric: str = "Tanimoto"):
        """

        :param probe_regid:
        :param probe_smiles:
        :param similarity_metric:
        :return:
        """
        probe = Molecule(probe_smiles, charges=charges)
        probe.generate_conformers(2*len(probe.exit_vectors))

        for conf_idx, vector in enumerate(probe.exit_vectors):
            alignment = AlignmentOneVector(probe_molecule=probe,
                                           query_molecule=self.query,
                                           query_exit_vectors=self.query.user_vectors[0],
                                           probe_exit_vectors=vector,
                                           probe_conformer_idx=2*conf_idx
                                           )
            alignment.align_and_score(similarity_metric=similarity_metric)

        best_conf_idx = max(probe.total_scores, key=probe.total_scores.get)
        # Need to floor divide because each vector alignment has two possible orientations, but these both correspond
        # to the same exit vector
        best_vector = probe.exit_vectors[best_conf_idx // 2]

        # Convert to SMILES
        best_smiles = self._vectors_to_dummies(probe, best_vector, update_mol=True)

        return [probe_regid, probe.total_scores[best_conf_idx], probe.shape_scores[best_conf_idx],
                probe.esp_scores[best_conf_idx], best_conf_idx, best_smiles, probe]

    def _vectors_to_dummies(self,
                            probe: Molecule,
                            vector_pair: list | tuple,
                            update_mol: bool = False) -> str:
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

        return probe.replace_hydrogens_with_dummy_atoms(hydrogens_to_replace, update_mol=update_mol)

    @staticmethod
    def _get_best_vector(vector_pairs: list, best_conf_idx: int) -> list:
        """
        Private method to return the best vector pair from a conformer index. If the index is odd, flip the order of
        the vectors in the vector pair, otherwise return the vector pair in the order that it is originally written
        :param vector_pairs:
        :return:
        """
        if best_conf_idx % 2 == 0:
            return vector_pairs[best_conf_idx // 2]
        else:
            return [vector_pairs[best_conf_idx // 2][1], vector_pairs[best_conf_idx // 2][0]]

    def align_and_score_orientation(self,
                                    probe: Molecule,
                                    probe_vector_pair,
                                    conformer_idx
                                    )-> None:
        """
        Aligns the probe molecule to the query molecule along the vectors specified by probe_vector_pair
        :param probe: probe molecule to align to query
        :param probe_vector_pair: probe vector pair to align to query vector pair
        :param conformer_idx: probe conformer to store alignment results
        :return: None
        """
        alignment = AlignmentTwoVector(query_molecule=self.query,
                                       probe_molecule=probe,
                                       query_exit_vectors=self.query.user_vectors,
                                       probe_exit_vectors=probe_vector_pair,
                                       probe_conformer_idx=conformer_idx,
                                       shape_weighting=self.shape_weight,
                                       esp_weighting=self.esp_weight
                                       )

        # Retrieve and store scores
        shape_sim, esp_sim = alignment.align_and_score()[0], alignment.align_and_score()[1]
        probe.shape_scores[conformer_idx] = shape_sim
        probe.esp_scores[conformer_idx] = esp_sim
        probe.total_scores[conformer_idx] = alignment.calculate_total_score(shape_sim, esp_sim)

        return None

    def initialise_probe_molecule(self,
                                  regid: str,
                                  num_of_vector_pairs: int,
                                  database_by_regid: dict
                                  )-> Molecule:
        """
        Initialise probe molecule with the appropriate number of conformers (twice the number of vector pairs)
        :param regid: the regid of the probe molecule
        :param num_of_vector_pairs: The number of vector pairs that correspond to the searched hash of the query
        molecule
        :return: instantiated Molecule
        """
        probe = Molecule(database_by_regid[regid]['smiles'],
                         charges=None if self.charge_type == 'Gasteiger' else database_by_regid[regid][self.charge_type]
                         )
        probe.generate_conformers(num_confs=2*num_of_vector_pairs)

        return probe
