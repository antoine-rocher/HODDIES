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
      install_requires=['numpy', 'scipy', 'mpytools', 'numba', 'yaml'],
      extras_require={'pycorr': ['pycorr @ git+https://github.com/cosmodesi/pycorr'], 'cosmoprimo': ['cosmoprimo @ git+https://github.com/cosmodesi/cosmoprimo'],
                      'fit_tools':['scikit-learn', 'idaes-pse'], 'abacusutils':['abacusutils[all]']}
      package_data={package_basename: ['default_HOD_parameters.yaml']},
      packages=find_packages()
)     
