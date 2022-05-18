from setuptools import setup

# from Cython.Build import cythonize
# from setuptools.extension import Extension

# extensions = [Extension('esp_gen', ['hcie/esp_gen.pyx'])]

setup(name='hcie',
      packages=['hcie'],
      include_package_data=True,
      package_data={'hcie': ['Data/*']},
      # ext_modules=cythonize(extensions, language_level='3', annotate=False),
      version='1.0.0',
      license='MIT',
      author='Matthew Holland',
      description='A package for computing aromatic heterocyclic bioisosteres of scaffold molecules')
