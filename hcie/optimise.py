"""
This script takes the SMILES string given as an input, and performs a DFT optimisation and a partial charge calculation,
generating output files which are stored in an appropriate directory.
"""

import autode
import csv


def optimise_geometry(smiles, name):
    """
    Optimise the geometry of the inputted query molecule
    :return: xyz file of optimised atomic co-ordinates
    """
    mol = autode.Molecule(name=name, smiles=smiles)
    mol.optimise(method=autode.methods.XTB())

    charges = calculate_charges(mol, name=name)

    print_charges(charges, name)
    mol.print_xyz_file()

    return None


def calculate_charges(molecule, name):
    """
    Calculate partial charges using Mulliken population analysis
    :param name: name of molecule, defaults to query
    :param molecule: instance of autode.Molecule class to calculate partial charges for
    :return: list of atomic partial charges
    """
    calc = autode.Calculation(name=f'{name}_calc', method=autode.methods.XTB(),
                              keywords=autode.Config.XTB.keywords.opt, molecule=molecule)

    calc.run()
    atomic_charges = calc.get_atomic_charges()

    return atomic_charges


def print_charges(charges, name):
    """
    print atomic charges to a csv file
    :param name: name of molecule, defaults to query
    :param charges: a list, ordered by atom index, of atomic partial charges
    :return: csv file of atomic partial charges
    """
    with open(f'{name}_charges.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(charges)

    return None
