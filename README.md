# The Heterocycle Isostere Explorer
## Introduction
This is a computational tool developed in the Brennan Group at the Centre for Medicines Discovery in the University 
of Oxford for the discovery of novel aromatic heterocyclic bioisosteres.

### Dependencies
* [Python](https://www.python.org/) > v. 3.7

The Python dependencies are listed in requirements.txt, and are best satisfied using a Conda installation ([miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) 
or [anaconda](https://docs.anaconda.com/anaconda/install/))

## Installation
To install **HCIE**, download the repository by clicking on the green <span style="color:green;">Code</span> button 
in the top right of this page. This will download a file onto your desktop (or specified location). Navigate to this 
folder in the command line (if you are unsure of this, follow instructions [here](https://www.wikihow.com/Change-Directories-in-Command-Prompt)).

It is highly advisable to create a new virtual environment for this package. If using conda, this can be achieved 
using
```
conda create env create -f environment.yml
```

Once all the packages have installed, activate the 
environment with
```
conda activate hcie_env
```
When in the folder, run the following to install HCIE into the virtual environment.
```
pip install .
``` 

## Usage

Once the package has been installed into the virtual environment, it can be run from any directory as long as the 
virtual environment is activated.

Searches are based on a SMILES string representation of the query molecule, with one or two attachment points 
indicated by dummy atoms. [This tool](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html) 
is very useful for generating SMILES expressions. Within a Python script (or session in the command line - 
```python```) run the following:

```aiignore
from hcie import VehicleSearch

search = VehicleSearch('<INSERT SMILES HERE>', name='<INSERT SEARCH NAME HERE>')
search.search()
```

This will start a search, which will take anywhere from 4 to 15 minutes depending on the search type. The results 
will be deposited in a directory SEARCHNAME_hcie_results.