from hcie.vehicle_search import VehicleSearch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles',
                        action='store',
                        type=str,
                        help='<Required> smiles string of molecule to search, with attachment points identified')
    parser.add_argument('-n', '--name',
                        action='store',
                        type=str,
                        help='name of molecule')

    return parser.parse_args()

def main():
    args = get_args()
    smiles = args.smiles
    name = args.name
    search = VehicleSearch(smiles, name=name)
    search.search()


if __name__ == '__main__':
    main()
