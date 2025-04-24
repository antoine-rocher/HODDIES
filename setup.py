import os
import sys
from setuptools import setup, find_packages

package_basename = 'HODDIES'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='Rocher Antoine',
      author_email='antoine.rocher@epfl.ch',
      description='Fast HOD code for small scale clustering analysis',
      license='GPLv3',
      url='https://github.com/antoine-rocher/HODDIES',
      install_requires=['matplotlib', 'numpy', 'mpytools', 'pycorr', 'numba', 'idaes-pse', 'scikit-learn'],
      #package_data={package_basename: ['*.mplstyle', 'data/*']},
      packages=find_packages()
)     