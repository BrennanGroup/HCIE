import argparse

from hcie import vehicle_search_parallel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles",
                        action='store',
                        type=str,
                        help="SMILES string of molecule to compare to VEHICLe"
                        )

    parser.add_argument("-n",
                        "--name",
                        action='store',
                        default='None',
                        type=str,
                        help="Name of query molecule"
                        )

    parser.add_argument("-o",
                        "--outputnum",
                        action='store',
                        type=int,
                        default=50,
                        help="Number of returned molecules to output to sdf file")

    return parser.parse_args()


args = get_args()
query_smiles = args.smiles
query_name = args.name
num_of_mols = args.outputnum
vehicle_search_parallel(query_smiles=query_smiles,
                        query_name=query_name,
                        num_of_output_mols=num_of_mols
                        )
