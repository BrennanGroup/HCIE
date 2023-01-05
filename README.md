# The Heterocyle Isostere Explorer
## *hcie*

---
This package uses the VEHICLe database of 24 867 aromatic heterocycles to find novel bio-isosteres of molecular scaffolds. It makes use of molecular shape and electrostatics (ShaEP) search algorithms to determine heterocycles of similarity to a query input, and can return the molecular coordinates of the identified isosteres aligned to the query molecule.


## Pre-Installation
Before installing and using HCIE you must have downloaded and installed [ShaEP](https://users.abo.fi/mivainio/shaep/download.php), and have added it to your bash profile. You must also have a python distribution. I recommend [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html) from the Anaconda (rather than Miniconda) distribution, which can be downloaded using instructions on this link. I recommend installing ShaEP onto your desktop, as that way the path is easy to find.

### Mac
#### ShaEP
Once you have installed ShaEP onto the desktop, the path should be `/Users/YOURUSERNAME/Desktop/ShaEP`, however if you have installed it elsewhere you will need to make a note of the path. Following installation, you need to add this path to your bash profile, so that HCIE can find the ShaEP executable. To do this open up a Terminal window and execute the following:
```
vi ~/.bash_profile
```
Once this has opened, scroll to the bottom of the inline document that opens, and then press the 'i' key, and enter to start a new line. On the new line write the following:
```
export PATH="/Users/YOURUSERNAME/Desktop:$PATH"
```
Then press the 'esc' key, type ':wq' and press enter. You should be returned to the terminal line automatically. Then type the following:
```
source ~/.bash_profile
```
and press enter. To check that this has worked, now type `which shaep`, if you get a path (names of folders separated by a forward slash) then all has gone well, if nothing happens then the path has not been added successfully.

#### HCIE
To install the latest version of HCIE, click on the green 'Code' button at the top of this page and press 'Download ZIP'. UnZip the folder and save the 'hcie-main' folder somewhere in your computer; for now let's assume its saved on Desktop. Now open up a terminal and change directory into the 'hcie-main' folder (`cd Desktop/hcie-main/`). Create the virtual environment by typing `conda env create -f environment.yml` in the working directory that you have installed hcie. Now activate the environment (`conda activate hcie`), and type `python setup.py install`. This should now have successfully installed the software onto your computer.

#### Environment
It is highly recommended that a new virtual environment is created for running HCIE. Using conda, this is achieved as follows:
```
conda create --name hcie_env
conda active hcie_env
conda install --file requirements.txt
python setup.py install
```

## Use
To use HCIE to search for potential isosteric replacements, a target molecule is required. This can either be initiated from a SMILES string, or a pre-determined geometry in an XYZ file. From within a python compiler (type `python` into the terminal line):

```
from hcie import Molecule

triazine = Molecule('c1ncncn1', name='triazine')
triazine.shaep()
```
This generates several files:

| File | Description |
| --- | --- |
| similarity.txt| This contains the similarity scores, as calculated by ShaEP, for all heterocycles in VEHICLe. It is ordered by the REGID of the VEHICLe molecules, and is somewhat overwhelming!|
| similarity_hits.txt | This contains the scores for the 20 most similar heterocycles in VEHICLe, as returned by ShaEP. They are listed in reverse order, with the most similar at the bottom of the list.|
| overlay.sdf | The coordinates of the best alignment of the top 20 most similar heterocycles (as defined in similarity_hits.txt) to the query molecule, and is useful for visualising the vectors to grow off. This can be visualised in PyMol using pymol_visualisation.py |
| pymol_visualisation.py | A python script to visualise the alignments in PyMol |
| triazine.mol2 | The MOL2 file of the optimised query molecule - these are the coordinates and charges that are used as the ShaEP query |

To visualise the alignments, open PyMol in the directory where the HCIE results are stored and then in the PyMol command line run:
```
run pymol_visualisation.py
```
