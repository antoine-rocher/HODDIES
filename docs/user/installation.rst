Installation
============

Requirements
------------

Strict requirements are:

  - numpy
  - scipy
  - yaml
  - `numba <https://numba.pydata.org/>`
  - `mpytools <https://github.com/cosmodesi/mpytools>`

Extra requirements are:

  - `pycorr <https://py2pcf.readthedocs.io/en/stable/>`_ for 2PCF computation
  - `cosmoprimo <https://cosmoprimo.readthedocs.io/en/latest/>`_ for cosmology tools
  - `abacusutils <https://abacusutils.readthedocs.io/en/latest/installation.html>`_ to load AbacusSummit simulations
  - `idaes-pse <ttps://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html>`` for inital sampling when performing HOD fits  
  - `scikit-learn <https://scikit-learn.org/stable/>`_ used for gaussian processes regression when performing HOD fits  

Pip Installation
----------------
Simply run:
::

    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES

Git installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to hack on the abacusutils source code, we recommend that you clone
the repo and install the package in pip "editable mode":

::

    $ git clone https://github.com/antoine-rocher/HODDIES.git
    $ cd HODDIES
    $ pip install -e .  # install all deps from current dir in editable mode


