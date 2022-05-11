"""
Contains functionality for searching a molecule against VEHICLe by various implemented methods
"""

from hcie.molecule import Molecule
from hcie.optimise import optimise_geometry
from hcie.mol2s import Mol2
from hcie.shaep import Shaep

import os


class Query(Molecule):

    def __init__(self,
                 smiles: str,
                 name: str = 'query'):
        super().__init__(smiles, name)

    @staticmethod
    def make_autode_directory():
        """
        Creates the directory for storing the geometry optimisation results
        :return: None
        """
        os.mkdir('autodE_outputs')
        return None

    @staticmethod
    def make_mol2_directory():
        """
        Creates the directory for storing the mol2 files
        :return: None
        """
        os.mkdir('Mol2')
        return None

    def optimise(self):
        """
        Optimises the geometry of the molecule and calculates partial charges
        :return: None
        """
        self.make_autode_directory()
        os.chdir('autodE_outputs')
        optimise_geometry(self.smiles, self.name)
        os.chdir('..')

        return None

    def create_mol2(self):
        """
        Creates mol2 file for query molecule
        :return: None
        """
        self.make_mol2_directory()
        os.chdir('Mol2')
        query_mol = Mol2(smiles=self.smiles, name=self.name)
        query_mol.print_mol2_file()
        os.chdir('..')

        return None

    @staticmethod
    def make_shaep_directory():
        """
        Creates directory for storing ShaEP outputs
        :return: None
        """
        os.mkdir('ShaEP')
        return None

    def search_shaep(self):
        """
        Performs a ShaEP search on the VEHICLe database for the query molecule
        :param args:
        :param kwargs:
        :return: None
        """
        self.make_shaep_directory()
        os.chdir('ShaEP')
        mol = Shaep(smiles=self.smiles, name=self.name)
        mol.shaep_search()
        os.chdir('..')

    def shaep(self):
        """
        Performs all the operations necessary to do a ShaEP search of VEHICLe on
        the instance of the query molecule.
        :param args:
        :param kwargs:
        :return:
        """
        self.optimise()
        self.create_mol2()
        self.search_shaep()

        return None
