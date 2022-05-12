# The Heterocyle Isostere Explorer
## *hcie*

---
This package uses the VEHICLe database of 24 867 aromatic heterocycles to find novel bio-isosteres of molecular scaffolds. It implements both ultrafast shape recognition (USR) and molecular shape and electrostatics (ShaEP) search algorithms, and can return the molecular coordinates of the identified isosteres aligned to the query molecule.


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

To install the latest version of HCIE, click on the green 'Code' button at the top of this page and press 'Download ZIP'. UnZip the folder and save the 'hcie-main' folder somewhere in your computer; for now let's assume its saved on Desktop. Now open up a terminal and change directory into the 'hcie-main' folder (`cd Desktop/hcie-main/`). Create the virtual environment by typing `conda env create -f environment.yml` in the working directory that you have installed hcie. Now activate the environment (`conda activate hcie`), and type `python setup.py install`. This should now have successfully installed the software onto your computer.

## Use
To use HCIE to search for potential isosteric replacements, a target molecule is required. This can either be initiated from a smiles string, or a .mol2 file. From within a python compiler (type `python` into the terminal line):

```
import hcie
from hcie.query import Query

target = Query(smiles='c1ccccc1', name='benzene')
target.shaep()
```
This will run a search of the VEHICLe database for benzene, and will create a new directory in the current working directory entitled 'benzene_hcie'.
