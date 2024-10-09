from setuptools import setup, find_packages

setup(
    name='hcie',
    version='0.1.0',
    description='A package for finding novel aromatic heterocyclic isosteres',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Matthew Holland',
    author_email='matthew.holland@cmd.ox.ac.uk',
    url='https://github.com/BrennanGroup/HCIE',
    packages=find_packages(),
    package_data={
        'HCIE_v2': ['data/*.json']
    },
    install_requires=[
        'numpy',
        'rdkit',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
