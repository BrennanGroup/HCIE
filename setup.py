from setuptools import setup

setup(
      name='hcie',
      packages=['hcie'],
      include_package_data=True,
      package_data={'hcie': ['Data/*']},
      version='1.0.0',
      license='MIT',
      author='Matthew Holland',
      author_email='matthew.holland@cmd.ox.ac.uk',
      description='A package for computing aromatic heterocyclic bioisosteres of scaffold molecules'
      )
