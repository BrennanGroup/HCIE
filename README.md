# The Heterocyle Isostere Explorer
## *hcie*

---
This package uses the VEHICLe database of 24 867 aromatic heterocycles to find novel bio-isosteres of molecular scaffolds. It aligns and scores each of the molecules in the VEHICLe database along the functionalisable bonds to the query molecule and the specified vector. The scores of all of the molecules in the database, and the alignments of the top 50 (by default, but more or less can be chosen by the user) alignments are also returned as an sdf file. 


## Pre-Installation
Before installing and using HCIE you must have a working Python 3 distribution on your machine.  [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html) from the Anaconda (rather than Miniconda) distribution, which can be downloaded using instructions on this link, is one such source of Python.

### Mac
#### HCIE
To install the latest version of HCIE, click on the green 'Code' button at the top of this page and press 'Download ZIP'. UnZip the folder and save the 'hcie-main' folder somewhere in your computer; for now let's assume its saved on Desktop. Now open up a terminal and change directory into the 'hcie-main' folder (`cd Desktop/hcie-main/`). Create the virtual environment as described below.

#### Environment
It is highly recommended that a new virtual environment is created for running HCIE. Using conda, this is achieved as follows:
```
conda create --name hcie_env
conda activate hcie_env
conda install --file requirements.txt
python setup.py install
```

## Use
To use HCIE to search for potential isosteric replacements, a target molecule is required. This can either be initiated from a SMILES string. The attachment vector should be indicated with '[R]'. SMILES strings can be drawn using ChemDraw on online tools such as [this one](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html). In a python file, or a python session in the command line, search using the following.

```
from hcie import vehicle_search_parallel

vehicle_search_parallel(query_smiles='[R]c1ccccn1',
                        query_name='2-pyridine')
```
This generates a new directory entitled 'hcie_results', into which two files are saved:

| File                   | Description                                                                                                                                                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2-pyridine.txt         | This contains all of the molecules in VEHICLe, listed in order of score against the query molecule. It includes the combined score, ESP score, Shape similarity score, and the conformer ID of the conformer of best alignment.  |
| 2-pyridine_results.sdf | The coordinates of the best alignment of the top 50 most similar heterocycles to the query molecule, and is useful for visualising the vectors to grow off. This can be opened in PyMol or any other molecular viewing software. |

