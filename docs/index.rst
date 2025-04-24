.. title:: HODDIES docs

**********************************
Welcome to HODDIES's documentation!
**********************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**HODDIES** is a Python toolkit to generate mock galaxy catalogs based on Halo Occupation Distribution (HOD) model. It also provide a tools to fit HOD models on data.   

A typical generation of a HOD-based mock is as simple as:

.. code-block:: python

    from HODDIES import HOD
    # Initiate the HOD instance with default parameter file parameters_HODdefaults.yaml
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
