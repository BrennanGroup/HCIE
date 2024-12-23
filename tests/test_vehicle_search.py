import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from hcie.molecule import Molecule
from hcie.alignment import AlignmentTwoVector
from hcie.vehicle_search import VehicleSearch


# Test constructor initialization
class TestOneVectorVehicleSearch(unittest.TestCase):
    def setUp(self):
        self.smiles = "[R]c1ccccn1"
        self.name = "ortho-pyridine"
        self.search_instance = VehicleSearch(self.smiles, self.name)
        # mock data for database_by_regid
        self.database_by_regid = {
            "S17": {"smiles": "O=c1[nH]cc[nH]1"},
            "S23050": {"smiles": "O=c1cnsn2scnon12"},
            "S408215": {"smiles": "Cc1cc2ncc(=O)[nH]n2c1N"},
            "S49": {"smiles": "c1ccncc1"},
        }

    def test_initialization(self):
        self.assertEqual(self.search_instance.smiles, self.smiles)
        self.assertEqual(self.search_instance.name, self.name)
        self.assertIsNone(self.search_instance.query_hash)
        self.assertEqual(self.search_instance.charge_type, "Gasteiger")
        self.assertEqual(self.search_instance.shape_weight, 0.5)
        self.assertEqual(self.search_instance.esp_weight, 0.5)
        self.assertIsNone(self.search_instance.query_charges)
        self.assertEqual(self.search_instance.search_type, "vector")

    def test_molecule_initialization(self):
        self.assertIsInstance(self.search_instance.query, Molecule)
        self.assertEqual(
            self.search_instance.query.smiles, self.smiles.replace("[R]", "[*]")
        )
        self.assertEqual(self.search_instance.query.name, self.name)
        self.assertEqual(self.search_instance.query.user_vectors, ((1, 0),))
        self.assertIsInstance(self.search_instance.query.coords, np.ndarray)

    @patch.object(VehicleSearch, "results_to_file")
    @patch.object(VehicleSearch, "align_and_score_vector_matches")
    @patch.object(VehicleSearch, "align_and_score_hash_matches_pooled")
    @patch("hcie.vehicle_search.load_database")
    def test_search(
        self,
        mock_load_database,
        mock_hash_matches,
        mock_vector_matches,
        mock_results_to_file,
    ):
        mock_load_database.return_value = self.database_by_regid
        mock_vector_matches.return_value = (["test_values"], {"mol": MagicMock()})
        self.search_instance.search()

        mock_vector_matches.assert_called_once()
        mock_hash_matches.assert_not_called()
        mock_results_to_file.assert_called_once()

    @patch.object(
        VehicleSearch, "align_and_score_probe_by_vector", return_value="mock_result"
    )
    def test_align_and_score_probe_by_vector_wrapper(
        self, mock_align_and_score_probe_by_vector
    ):
        args = ("RegID", "SMILES")
        kwargs = {"similarity_metric": "Tanimoto"}

        result = self.search_instance.align_and_score_probe_by_vector_wrapper(
            (args, kwargs)
        )

        mock_align_and_score_probe_by_vector.assert_called_once_with(*args, **kwargs)

        self.assertEqual(result, "mock_result")

    def test_align_and_score_vector_matches(self):
        mock_result = [
            ["S17", 1.4, 0.8, 0.6, 1, "s17_smiles", MagicMock()],
            ["S23050", 0.9, 0.6, 0.3, 0, "S23050_smiles", MagicMock()],
            ["S408215", 0.6, 0.3, 0.3, 2, "S408215_smiles", MagicMock()],
            ["S49", 2.0, 1.0, 1.0, 1, "S49_smiles", MagicMock()],
        ]

        with patch("multiprocessing.Pool") as mock_pool:
            mock_pool.return_value.__enter__.return_value.imap_unordered.return_value = (
                mock_result
            )
            results, processed_mols = (
                self.search_instance.align_and_score_vector_matches(
                    self.database_by_regid
                )
            )

            self.assertEqual(len(results), 4)
            self.assertEqual(results[0][0], "S49")
            self.assertEqual(results[0][1], 2.0)
            self.assertEqual(results[1][0], "S17")
            self.assertEqual(results[1][1], 1.4)
            self.assertEqual(results[2][0], "S23050")
            self.assertEqual(results[2][5], "S23050_smiles")

            self.assertEqual(len(processed_mols), 4)
            self.assertEqual(processed_mols["S49"], mock_result[3][-1])
            self.assertEqual(processed_mols["S17"], mock_result[0][-1])
            self.assertEqual(processed_mols["S23050"], mock_result[1][-1])
            self.assertEqual(processed_mols["S408215"], mock_result[2][-1])

    def test_generate_single_vector_tasks(self):
        task_args = self.search_instance.generate_single_vector_tasks(
            self.database_by_regid
        )

        self.assertEqual(len(task_args), 4)
        self.assertEqual(task_args[0][0][0], "S17")
        self.assertEqual(task_args[0][0][1], "O=c1[nH]cc[nH]1")
        self.assertIsInstance(task_args[0], tuple)
        self.assertIsInstance(task_args[0][1], dict)
        self.assertEqual(task_args[0][1]["similarity_metric"], "Tanimoto")

        with patch.object(self.search_instance, "charge_type", "other_charges"):
            with self.assertRaises(ValueError):
                task_args = self.search_instance.generate_single_vector_tasks(
                    self.database_by_regid
                )

    def test_align_and_score_probe_by_vector(self):
        # Test that an alignment against itself returns a near perfect score
        result = self.search_instance.align_and_score_probe_by_vector(
            probe_regid="S49", probe_smiles=self.smiles
        )

        self.assertEqual(len(result[-1].total_scores), 10)
        self.assertEqual(len(result[-1].shape_scores), 10)
        self.assertEqual(len(result[-1].esp_scores), 10)

        self.assertTrue(all(v is not None for v in result[-1].total_scores.values()))
        self.assertTrue(all(v is not None for v in result[-1].shape_scores.values()))
        self.assertTrue(all(v is not None for v in result[-1].esp_scores.values()))

        self.assertEqual(len(result), 7)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], int)
        self.assertIsInstance(result[5], str)
        self.assertIsInstance(result[6], Molecule)

        self.assertAlmostEqual(result[1], 2, 1)
        self.assertAlmostEqual(result[2], 1, 1)
        self.assertAlmostEqual(result[3], 1, 1)


class TestTwoVectorVehicleSearch(unittest.TestCase):
    def setUp(self):
        self.smiles = "[*:1]c1cccc2c1ccn2[*:2]"
        self.name = "indole"
        self.search_instance = VehicleSearch(self.smiles, self.name)
        # mock data for database_by_regid
        self.database_by_regid = test_dict

    def test_initialization(self):
        self.assertEqual(self.search_instance.smiles, self.smiles)
        self.assertEqual(self.search_instance.name, self.name)
        self.assertIsInstance(self.search_instance.query_hash, str)
        self.assertEqual(self.search_instance.query_hash, "00111011")
        self.assertEqual(self.search_instance.charge_type, "Gasteiger")
        self.assertEqual(self.search_instance.search_type, "hash")

    @patch.object(VehicleSearch, "results_to_file")
    @patch.object(VehicleSearch, "get_exit_vectors_for_hash_matches")
    @patch.object(VehicleSearch, "align_and_score_vector_matches")
    @patch.object(VehicleSearch, "align_and_score_hash_matches_pooled")
    @patch("hcie.vehicle_search.load_database")
    def test_search(
        self,
        mock_load_database,
        mock_hash_matches,
        mock_vector_matches,
        mock_exit_vectors,
        mock_results_to_file,
    ):
        mock_load_database.return_value = self.database_by_regid
        mock_hash_matches.return_value = (["test_values"], {"mol": MagicMock()})
        self.search_instance.search()

        mock_hash_matches.assert_called_once()
        mock_exit_vectors.assert_called_once()
        mock_results_to_file.assert_called_once()
        mock_vector_matches.assert_not_called()

        with patch.object(self.search_instance, "search_type", "unrecognised_type"):
            with self.assertRaises(ValueError):
                self.search_instance.search()

    def test_get_exit_vectors_for_hash_matches(self):
        test_matches = ["S290", "S231401"]
        with patch.object(self.search_instance, "hash_matches", test_matches):
            vectors = self.search_instance.get_exit_vectors_for_hash_matches(
                self.database_by_regid
            )
            self.assertIsInstance(vectors, dict)
            self.assertEqual(len(vectors), len(test_matches))
            self.assertTrue(
                not (any(key for key in vectors.keys() if key not in test_matches))
            )

    @patch(
        "hcie.vehicle_search.vehicle_by_hash",
        {
            "00111011": ["S17", "S49", "S278"],
            "00010011": ["S23050"],
            "00000010": ["S408215"],
        },
    )
    def test_search_vehicle_by_hash(self):
        hash_matches = self.search_instance.search_vehicle_by_hash()
        self.assertIsInstance(hash_matches, list)
        self.assertEqual(len(hash_matches), 3)

        with patch.object(self.search_instance, "query_hash", "11111111"):
            with self.assertRaises(KeyError):
                self.search_instance.search_vehicle_by_hash()

    def test_align_and_score_vehicle_molecule(self):
        vector_pairs = [
            vector["vectors"]
            for vector in self.database_by_regid["S290"]["exit_vectors"]["00111011"]
        ]
        result = self.search_instance.align_and_score_vehicle_molecule(
            regid="S290",
            vector_pairs=vector_pairs,
            database_by_regid=self.database_by_regid,
        )

        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result[1], 2, 1)
        self.assertIsInstance(result[-1], Molecule)

    @patch.object(
        VehicleSearch, "align_and_score_vehicle_molecule", return_value="mock_return"
    )
    def test_align_and_score_molecule_wrapper(self, mock_align_and_score):
        args = ("RegID", [1, 2], self.database_by_regid)

        result = self.search_instance.align_and_score_molecule_wrapper(args)

        mock_align_and_score.assert_called_once_with(*args)

        self.assertEqual(result, "mock_return")

    def test_align_and_score_hash_matches_pooled(self):
        mock_result = [
            ["S17", 1.4, 0.8, 0.6, 1, "s17_smiles", MagicMock()],
            ["S23050", 0.9, 0.6, 0.3, 0, "S23050_smiles", MagicMock()],
            ["S408215", 0.6, 0.3, 0.3, 2, "S408215_smiles", MagicMock()],
            ["S49", 2.0, 1.0, 1.0, 1, "S49_smiles", MagicMock()],
        ]

        with patch("multiprocessing.Pool") as mock_pool:
            mock_pool.return_value.__enter__.return_value.imap_unordered.return_value = (
                mock_result
            )
            results, processed_mols = (
                self.search_instance.align_and_score_vector_matches(
                    self.database_by_regid
                )
            )

            self.assertEqual(len(results), 4)
            self.assertEqual(results[0][0], "S49")
            self.assertEqual(results[0][1], 2.0)
            self.assertEqual(results[1][0], "S17")
            self.assertEqual(results[1][1], 1.4)
            self.assertEqual(results[2][0], "S23050")
            self.assertEqual(results[2][5], "S23050_smiles")

            self.assertEqual(len(processed_mols), 4)
            self.assertEqual(processed_mols["S49"], mock_result[3][-1])
            self.assertEqual(processed_mols["S17"], mock_result[0][-1])
            self.assertEqual(processed_mols["S23050"], mock_result[1][-1])
            self.assertEqual(processed_mols["S408215"], mock_result[2][-1])

    @patch.object(AlignmentTwoVector, "align_and_score")
    def test_align_and_score_orientation_with_mock_probe(self, mock_alignment_class):
        mock_alignment_class.return_value = (0.8, 0.6)

        # Create a mock probe
        mock_probe = MagicMock()
        mock_probe.shape_scores = {}
        mock_probe.esp_scores = {}
        mock_probe.total_scores = {}
        mock_probe.coords = MagicMock()
        mock_probe.centroid = MagicMock()

        # Inputs
        probe_vector_pair = [(5, 6), (7, 8)]
        conformer_idx = 1

        # Call the method under test with the mock probe
        self.search_instance.align_and_score_orientation(
            probe=mock_probe,
            probe_vector_pair=probe_vector_pair,
            conformer_idx=conformer_idx,
        )

        self.assertEqual(mock_probe.shape_scores[conformer_idx], 0.8)
        self.assertEqual(mock_probe.esp_scores[conformer_idx], 0.6)
        self.assertEqual(mock_probe.total_scores[conformer_idx], 1.4)

    def test_initialise_probe_molecule(self):
        molecule = self.search_instance.initialise_probe_molecule(
            "S17", 6, self.database_by_regid
        )

        self.assertIsInstance(molecule, Molecule)
        self.assertEqual(molecule.mol.GetNumConformers(), 13)

        with self.assertRaises(KeyError):
            molecule = self.search_instance.initialise_probe_molecule(
                "SR", 6, self.database_by_regid
            )


test_dict = {
    "S17": {
        "exit_vectors": {
            "00000010": [
                {
                    "vectors": [[2, 6], [3, 7]],
                    "distance": 1.371459705774545,
                    "angles": {
                        "av": 69.46351784934379,
                        "a1": 122.15802097552107,
                        "a2": 127.30549687382272,
                    },
                },
                {
                    "vectors": [[3, 7], [4, 8]],
                    "distance": 1.339740875685228,
                    "angles": {
                        "av": 78.6336814262774,
                        "a1": 129.31681356004668,
                        "a2": 129.31686786623072,
                    },
                },
                {
                    "vectors": [[4, 8], [5, 9]],
                    "distance": 1.371459792676174,
                    "angles": {
                        "av": 69.46356360308513,
                        "a1": 122.15807085575462,
                        "a2": 127.30549274733052,
                    },
                },
            ],
            "00001100": [
                {
                    "vectors": [[2, 6], [4, 8]],
                    "distance": 2.200764892414422,
                    "angles": {
                        "av": 148.09719927169073,
                        "a1": 162.56061880124884,
                        "a2": 165.5365804704419,
                    },
                },
                {
                    "vectors": [[2, 6], [5, 9]],
                    "distance": 2.2112220448627027,
                    "angles": {
                        "av": 142.43923711249022,
                        "a1": 161.2196130206223,
                        "a2": 161.21962409186793,
                    },
                },
                {
                    "vectors": [[3, 7], [5, 9]],
                    "distance": 2.200765024536974,
                    "angles": {
                        "av": 148.0972450269001,
                        "a1": 162.56061119285866,
                        "a2": 165.53663383404145,
                    },
                },
            ],
        },
        "smiles": "O=c1[nH]cc[nH]1",
        "num_vectors": 4,
    },
    "S23050": {
        "exit_vectors": {
            "01100011": [
                {
                    "vectors": [[2, 11], [7, 12]],
                    "distance": 4.9848418450885426,
                    "angles": {
                        "av": 126.94542566754231,
                        "a1": 151.33592504580517,
                        "a2": 155.60950062173714,
                    },
                }
            ]
        },
        "smiles": "O=c1cnsn2scnon12",
        "num_vectors": 2,
    },
    "S49": {
        "exit_vectors": {
            "00000010": [
                {
                    "vectors": [[0, 6], [1, 7]],
                    "distance": 1.3909630407955589,
                    "angles": {
                        "av": 61.457646814699785,
                        "a1": 120.34168541590441,
                        "a2": 121.11596139879538,
                    },
                },
                {
                    "vectors": [[0, 6], [5, 10]],
                    "distance": 1.3909629666118601,
                    "angles": {
                        "av": 61.45764699592711,
                        "a1": 120.34165674138066,
                        "a2": 121.11599025454645,
                    },
                },
                {
                    "vectors": [[1, 7], [2, 8]],
                    "distance": 1.3858071499103881,
                    "angles": {
                        "av": 61.658812564604176,
                        "a1": 120.64977234500293,
                        "a2": 121.00904021960125,
                    },
                },
                {
                    "vectors": [[4, 9], [5, 10]],
                    "distance": 1.385806984010296,
                    "angles": {
                        "av": 61.6587251056164,
                        "a1": 120.64973566929991,
                        "a2": 121.00898943631648,
                    },
                },
            ],
            "00010011": [
                {
                    "vectors": [[0, 6], [2, 8]],
                    "distance": 2.383076826712463,
                    "angles": {
                        "av": 123.11645937093843,
                        "a1": 151.16092452154362,
                        "a2": 151.9555348493948,
                    },
                },
                {
                    "vectors": [[0, 6], [4, 9]],
                    "distance": 2.383076717979126,
                    "angles": {
                        "av": 123.11637209409011,
                        "a1": 151.16089081023472,
                        "a2": 151.9554812838554,
                    },
                },
                {
                    "vectors": [[1, 7], [5, 10]],
                    "distance": 2.4008810971089014,
                    "angles": {
                        "av": 122.91529380938397,
                        "a1": 151.4576315819899,
                        "a2": 151.45766222739408,
                    },
                },
                {
                    "vectors": [[2, 8], [4, 9]],
                    "distance": 2.2989611073148057,
                    "angles": {
                        "av": 113.76716852378692,
                        "a1": 146.8835598533454,
                        "a2": 146.88360867044153,
                    },
                },
            ],
            "00011101": [
                {
                    "vectors": [[1, 7], [4, 9]],
                    "distance": 2.7276351711538704,
                    "angles": {
                        "av": 175.42598094687193,
                        "a1": 177.39556414593454,
                        "a2": 178.0304168009374,
                    },
                },
                {
                    "vectors": [[2, 8], [5, 10]],
                    "distance": 2.727635121770651,
                    "angles": {
                        "av": 175.42589353649058,
                        "a1": 177.39551186719555,
                        "a2": 178.03038166929502,
                    },
                },
            ],
        },
        "smiles": "c1ccncc1",
        "num_vectors": 5,
    },
    "S408215": {
        "smiles": "Cc1cc2ncc(=O)[nH]n2c1N",
        "num_vectors": 3,
        "exit_vectors": {
            "00111010": [
                {
                    "vectors": [[2, 15], [5, 16]],
                    "distance": 3.592093786960354,
                    "angles": {
                        "av": 81.91610600309166,
                        "a1": 113.8575796272663,
                        "a2": 148.05852637582535,
                    },
                }
            ],
            "00110011": [
                {
                    "vectors": [[2, 15], [8, 17]],
                    "distance": 3.4899066971829393,
                    "angles": {
                        "av": 99.55887205980017,
                        "a1": 125.26648737339065,
                        "a2": 154.29238468640952,
                    },
                }
            ],
            "00010011": [
                {
                    "vectors": [[5, 16], [8, 17]],
                    "distance": 2.4493589871726775,
                    "angles": {
                        "av": 111.49663135295435,
                        "a1": 144.3278275651315,
                        "a2": 147.16880378782284,
                    },
                }
            ],
        },
    },
    "S278": {
        "exit_vectors": {
            "00000010": [
                {
                    "vectors": [[0, 8], [1, 9]],
                    "distance": 1.417700832209402,
                    "angles": {
                        "av": 75.10030186806218,
                        "a1": 125.46165286662504,
                        "a2": 129.63864900143713,
                    },
                },
                {
                    "vectors": [[0, 8], [7, 10]],
                    "distance": 1.3962163046718148,
                    "angles": {
                        "av": 71.21598774172838,
                        "a1": 119.08148448673612,
                        "a2": 132.13450325499227,
                    },
                },
            ],
            "00010100": [
                {
                    "vectors": [[1, 9], [7, 10]],
                    "distance": 2.3229401139010775,
                    "angles": {
                        "av": 146.31628960248506,
                        "a1": 159.52263567086496,
                        "a2": 166.7936539316201,
                    },
                }
            ],
        },
        "smiles": "c1cc2nonc2[nH]1",
        "num_vectors": 3,
    },
    "S231401": {
        "smiles": "Clc1nn2[nH]cnoc-2c[nH]1",
        "num_vectors": 4,
        "exit_vectors": {
            "00000010": [
                {
                    "vectors": [[4, 11], [5, 12]],
                    "distance": 1.3644080867160155,
                    "angles": {
                        "av": 51.178691894872756,
                        "a1": 112.5557777122135,
                        "a2": 118.62291418265926,
                    },
                },
                {
                    "vectors": [[9, 13], [10, 14]],
                    "distance": 1.3739091893682207,
                    "angles": {
                        "av": 57.86076139826086,
                        "a1": 117.59197199112391,
                        "a2": 120.26878940713695,
                    },
                },
            ],
            "00111011": [
                {
                    "vectors": [[4, 11], [9, 13]],
                    "distance": 3.584914105658096,
                    "angles": {
                        "av": 97.67396713050846,
                        "a1": 132.98588327099748,
                        "a2": 144.68808385951098,
                    },
                }
            ],
            "01001011": [
                {
                    "vectors": [[4, 11], [10, 14]],
                    "distance": 4.007022266966218,
                    "angles": {
                        "av": 111.53767688813757,
                        "a1": 115.12454442449949,
                        "a2": 176.41313246363808,
                    },
                }
            ],
            "01000011": [
                {
                    "vectors": [[5, 12], [9, 13]],
                    "distance": 3.9889105320394065,
                    "angles": {
                        "av": 116.92073082282474,
                        "a1": 125.1639299071482,
                        "a2": 171.75680091567654,
                    },
                }
            ],
            "01100100": [
                {
                    "vectors": [[5, 12], [10, 14]],
                    "distance": 4.776032736775069,
                    "angles": {
                        "av": 147.06339826150318,
                        "a1": 158.85730753611443,
                        "a2": 168.20609072538875,
                    },
                }
            ],
        },
    },
    "S290": {
        "exit_vectors": {
            "00000010": [
                {
                    "vectors": [[0, 9], [1, 10]],
                    "distance": 1.3934572593117904,
                    "angles": {
                        "av": 59.33706005004527,
                        "a1": 119.53222625636744,
                        "a2": 119.80483379367783,
                    },
                },
                {
                    "vectors": [[0, 9], [8, 15]],
                    "distance": 1.3998432203397644,
                    "angles": {
                        "av": 59.776715550613616,
                        "a1": 119.52226760375918,
                        "a2": 120.25444794685444,
                    },
                },
                {
                    "vectors": [[1, 10], [2, 11]],
                    "distance": 1.3983878511919492,
                    "angles": {
                        "av": 60.256296321829055,
                        "a1": 119.4100862452624,
                        "a2": 120.84621007656665,
                    },
                },
                {
                    "vectors": [[4, 12], [5, 13]],
                    "distance": 1.3698987331201757,
                    "angles": {
                        "av": 66.1441422678523,
                        "a1": 120.8657425590921,
                        "a2": 125.2783997087602,
                    },
                },
                {
                    "vectors": [[5, 13], [6, 14]],
                    "distance": 1.3762824780938,
                    "angles": {
                        "av": 77.02677747575673,
                        "a1": 125.68066658586979,
                        "a2": 131.34611088988694,
                    },
                },
            ],
            "00010011": [
                {
                    "vectors": [[0, 9], [2, 11]],
                    "distance": 2.430591281320205,
                    "angles": {
                        "av": 119.5933563565224,
                        "a1": 149.33317238396648,
                        "a2": 150.26018397255592,
                    },
                },
                {
                    "vectors": [[1, 10], [8, 15]],
                    "distance": 2.4272308160441325,
                    "angles": {
                        "av": 119.11377558138662,
                        "a1": 149.27038069028598,
                        "a2": 149.84339489110064,
                    },
                },
            ],
            "01001011": [
                {
                    "vectors": [[0, 9], [4, 12]],
                    "distance": 4.122316091836927,
                    "angles": {
                        "av": 128.69084362607043,
                        "a1": 132.3241655013074,
                        "a2": 176.36667812476304,
                    },
                },
                {
                    "vectors": [[1, 10], [6, 14]],
                    "distance": 4.1934960963743775,
                    "angles": {
                        "av": 131.7435213050873,
                        "a1": 135.97608717148853,
                        "a2": 175.76743413359878,
                    },
                },
            ],
            "01011100": [
                {
                    "vectors": [[0, 9], [5, 13]],
                    "distance": 4.614640423561289,
                    "angles": {
                        "av": 157.89837039652187,
                        "a1": 159.51249012572626,
                        "a2": 178.3858802707956,
                    },
                },
                {
                    "vectors": [[1, 10], [5, 13]],
                    "distance": 4.599398720833945,
                    "angles": {
                        "av": 142.7645695362215,
                        "a1": 158.54608014267882,
                        "a2": 164.21848939354268,
                    },
                },
            ],
            "01000010": [
                {
                    "vectors": [[0, 9], [6, 14]],
                    "distance": 3.81217787759677,
                    "angles": {
                        "av": 80.87159292595585,
                        "a1": 116.67888566367246,
                        "a2": 144.1927072622834,
                    },
                }
            ],
            "00111010": [
                {
                    "vectors": [[1, 10], [4, 12]],
                    "distance": 3.7239871473084243,
                    "angles": {
                        "av": 76.6204272731277,
                        "a1": 112.70163252742755,
                        "a2": 143.91879474570015,
                    },
                },
                {
                    "vectors": [[2, 11], [5, 13]],
                    "distance": 3.6238643858827624,
                    "angles": {
                        "av": 82.508273228438,
                        "a1": 112.38624950614302,
                        "a2": 150.12202372229498,
                    },
                },
            ],
            "00011001": [
                {
                    "vectors": [[2, 11], [4, 12]],
                    "distance": 2.5192927863291574,
                    "angles": {
                        "av": 16.36413096058314,
                        "a1": 96.97494888796444,
                        "a2": 99.3891820726187,
                    },
                },
                {
                    "vectors": [[6, 14], [8, 15]],
                    "distance": 2.6064424333320586,
                    "angles": {
                        "av": 21.0948773830618,
                        "a1": 97.37020247064686,
                        "a2": 103.72467491241494,
                    },
                },
            ],
            "00111011": [
                {
                    "vectors": [[2, 11], [6, 14]],
                    "distance": 3.6115513949381635,
                    "angles": {
                        "av": 109.09699305950122,
                        "a1": 134.31602188175933,
                        "a2": 154.7809711777419,
                    },
                },
                {
                    "vectors": [[4, 12], [8, 15]],
                    "distance": 3.555419215040286,
                    "angles": {
                        "av": 107.40539445139709,
                        "a1": 135.8355957894826,
                        "a2": 151.5697986619145,
                    },
                },
                {
                    "vectors": [[5, 13], [8, 15]],
                    "distance": 3.654600343997991,
                    "angles": {
                        "av": 98.12165485791377,
                        "a1": 113.9862106908729,
                        "a2": 164.13544416704087,
                    },
                },
            ],
            "00100101": [
                {
                    "vectors": [[2, 11], [8, 15]],
                    "distance": 2.8290038612629003,
                    "angles": {
                        "av": 179.37007138521534,
                        "a1": 179.45861952095825,
                        "a2": 179.9114518642571,
                    },
                }
            ],
            "00001100": [
                {
                    "vectors": [[4, 12], [6, 14]],
                    "distance": 2.218722443754156,
                    "angles": {
                        "av": 143.1709197416871,
                        "a1": 161.4814706694504,
                        "a2": 161.6894490722367,
                    },
                }
            ],
        },
        "smiles": "c1ccc2[nH]ccc2c1",
        "num_vectors": 7,
    },
}
