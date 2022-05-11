"""
Extends Molecule class to run ShaEP search on the VEHICLe database
"""

from hcie.molecule import Molecule
import os
import shutil
import subprocess
import pkg_resources

# Set path to package data files referenced within the code
VEHICLE_MOL2 = pkg_resources.resource_filename('hcie', 'Data/vehicle_dft.mol2')


class Shaep(Molecule):
    """
    Extends molecule class to run a ShaEP search on the VEHICLe database
    """

    def __init__(self,
                 smiles: str,
                 name: str = 'query',
                 max_hits: int = 20,
                 output_queries: bool = True):
        super().__init__(smiles=smiles, name=name)
        self.max_hits = max_hits
        self.output_queries = output_queries

    @property
    def query_file(self):
        """
        name of query file
        :return: name of query file
        """
        return f'{self.name}.mol2'

    def get_query_path(self):
        """
        Finds the relative path to the query file in the Mol2 directory
        :return: path to query file
        """
        if os.path.basename(os.getcwd()) != 'ShaEP':
            os.chdir('ShaEP')
        else:
            query_path = os.path.join('..', 'Mol2', f'{self.name}.mol2')

        return query_path

    @staticmethod
    def get_shaep_path():
        """
        Tests that the ShaEP executable is installed, and returns its path if so
        :return: Path to ShaEP executable
        """
        if shutil.which('shaep') is None:
            raise Exception('ShaEP not installed')
        else:
            shaep_path = shutil.which('shaep')

        return shaep_path

    def run_search(self, shaep_path, query_file):
        """
        Runs a ShaEP search of the query molecule in the VEHICLe database
        :param shaep_path: path to ShaEP executable
        :param query_file: path to query molecule Mol2 file
        :return: None
        """
        if self.output_queries is True:
            subprocess.run([shaep_path, '--maxhits', str(self.max_hits), '-q', query_file, '--output-file', 'similarity.txt',
                            '--structures',
                            'overlay.sdf', '--outputQuery', VEHICLE_MOL2])
        else:
            subprocess.run([shaep_path, '--maxhits', str(self.max_hits), '-q', query_file, '--output-file', 'similarity.txt',
                            '--structures',
                            'overlay.sdf', VEHICLE_MOL2])

        return None

    def shaep_search(self):
        """
        Searches the VEHICLe database for isosteres using the ShaEP tool written by Dr Mikko Vainio and co-workers at
        Abo Akademi University in Finland in 2009 (Vainio et al, 2009).
        """
        shaep_path = self.get_shaep_path()
        query_file = self.get_query_path()

        self.run_search(shaep_path, query_file)

        return None
