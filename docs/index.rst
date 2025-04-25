.. title:: HODDIES docs

**********************************
Welcome to HODDIES's documentation!
**********************************
.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/installation
  api/api

.. toctree::
  :hidden:
************
Introduction
************

**HODDIES** is a Python toolkit to generate mock galaxy catalogs based on Halo Occupation Distribution (HOD) models. It also provide a tools to fit HOD models on data.   

A typical generation of a HOD-based mock is as simple as:

.. code-block:: python

    from HODDIES import HOD
    # Initiate the HOD instance with default parameter file parameters_HODdefaults.yaml
    # Using path to catalog

    path_to_halo_catalog = '/path/to/halo_catalog' # catalog type can be fits, h5, asdf
    HOD_obj= HOD(hcat_file='path_to_halo_catalog')

    # Using a preloaded catalog. 
    # This catalog need to be a dictonary / structured array or mpy Catalog.
    # Colunms names should be at least:
    # ['x', 'y', 'z', 'vx', 'vy', 'vz','Mh', 'Rh', 'Rs', 'c', 'Vrms', 'halo_id']  

    HOD_obj= HOD(hcat_file=mycatalog) 

    # Using root AbacusSummit simulation path, example at NERSC

    HOD_obj= HOD(path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')

    # Generate the mock catalog

    mock_cat = HOD_obj.make_mock_cat(fix_seed=None) 

    # Compute clustering statistics

    rp, wp = HOD_obj.get_wp(cats, tracers='ELG')
    s, xi = HOD_obj.get_2PCF(cats, tracers='ELG')

Example notebooks are provided in :root:`nb`. 



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
