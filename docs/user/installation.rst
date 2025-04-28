Installation
============

Requirements
------------

Strict requirements are:

  - ``numpy``
  - ``scipy``
  - ``yaml``
  - `numba <https://numba.pydata.org/>`_
  - `mpytools <https://github.com/cosmodesi/mpytools>`_

Extra requirements are:

  - `pycorr <https://py2pcf.readthedocs.io/en/stable/>`_ for 2PCF computation (installed without Corrfunc )
  - `cosmoprimo <https://cosmoprimo.readthedocs.io/en/latest/>`_ for cosmology tools
  - `abacusutils <https://abacusutils.readthedocs.io/en/latest/installation.html>`_ to load AbacusSummit simulations
  - `idaes-pse <https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html>`_ for inital sampling when performing HOD fits  
  - `scikit-learn <https://scikit-learn.org/stable/>`_ used for gaussian processes regression when performing HOD fits  
  - `colossus <https://bdiemer.bitbucket.io/colossus/index.html>`_ for mass concentration relations

Pip Installation
----------------
Simply run:
::

    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES

This will install dependencies, to generate mocks. To install all extra requirements use:
::

    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[all]


To install only a part of extra requirements you can do:
::

    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[cosmodesi]    # install ['pycorr', 'cosmoprimo']
    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[fit_tools]    # install ['scikit-learn','emcee','zeus','idaes-pse']
    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[colossus]     # install ['colossus']
    $ python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[abacusutils]  # install ['abacusutils']

``Pycorr`` and ``Corrfunc`` installation
----------------------------------------


HODDIES provide two-point correlation measurement based on ``pycorr`` which use a specific branch of Corrfunc. ``pycorr`` is installed as an extra dependency in **HODDIES** without two-point counter engine, so fairly unusable. ``pycorr`` currently use a specific branch of Corrfunc, located `here <https://github.com/cosmodesi/Corrfunc/tree/desi>`_. Details on ``pycorr`` installation can be found `here <https://github.com/cosmodesi/Corrfunc/tree/desi>`_. 

To install ``pycorr``with ``Corrfunc``, first, uninstall previous ``Corrfunc`` version (if any):
::

    $ pip uninstall Corrfunc


To install ``Corrfunc`` if ``pycorr`` is already install :
::

    $ python -m pip install git+https://github.com/adematti/Corrfunc@desi

To install ``Corrfunc`` and ``pycorr`` if both are not install:
::

    $ python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]


Git installation
----------------
::

    $ git clone https://github.com/antoine-rocher/HODDIES.git
    $ cd HODDIES
    $ pip install -e .[all]  # install all deps from current dir in editable mode


