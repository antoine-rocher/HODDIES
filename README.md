# HODDIES

**HODDIES** is a Python toolkit to generate mock galaxy catalogs based on Halo Occupation Distribution (HOD) model. It also provides tools to fit HOD models on data. 

![HOD cartoon](https://github.com/antoine-rocher/HODDIES/blob/main/HOD_cartoon.png)


## Documentation

Documentation is hosted on Read the Docs, [HODDIES docs](https://hoddies.readthedocs.io).

### Examples

An example can using a custom halo catlog. The code is initialized using the default [paramerter file](https://github.com/antoine-rocher/HODDIES/blob/main/HODDIES/default_HOD_parameters.yaml)
(pseudo-code, for an example with all variables defined see [this notebook](https://github.com/antoine-rocher/HODDIES/blob/main/nb/basic_HOD_examples.ipynb)):
```
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
```

One can also plot the HOD distribution and the Halo mass function (HMF) of the output mock catalog:
```
#Plot HOD

HOD_obj.HOD_plot()

#Plot Halo mass function 

HOD_obj.plot_HMF(mock_cat, show_sat=True)

```

Example notebooks are provided in directory [nb/](https://github.com/antoine-rocher/HODDIES/tree/main/nb).


## Requirements

Strict requirements are:

  - numpy
  - scipy
  - yaml
  - [numba](https://numba.pydata.org/)
  - [mpytools](https://github.com/cosmodesi/mpytools)
  

Extra requirements are:

  - [pycorr](https://py2pcf.readthedocs.io/en/stable/) for 2PCF computation
  - [cosmoprimo](https://cosmoprimo.readthedocs.io/en/latest/) for cosmology tools
  - [abacusutils](https://abacusutils.readthedocs.io/en/latest/installation.html) to load AbacusSummit simulations
  - [idaes-pse](https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html) for inital sampling when performing HOD fits  
  - [scikit-learn](https://scikit-learn.org/stable/) used for gaussian processes regression when performing HOD fits  

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/antoine-rocher/HODDIES#egg=HODDIES[all]
```

### git

First:
```
git clone https://github.com/antoine-rocher/HODDIES.git
```
To install the code:
```
python setup.py install --user
```

## License

**HODDIES** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/antoine-rocher/HODDIES/blob/main/LICENSE).

## Credits

Arnaud de Mattia for useful tools from [cosmodesi](https://github.com/cosmodesi)
 