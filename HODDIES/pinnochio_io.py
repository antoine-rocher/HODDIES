from mpytools import Catalog
import os
import numpy as np
import socket


def get_concentration(M, z, cosmo, cmrelation='diemer19', mdef='200c'):
    from colossus.halo import concentration
    from colossus.cosmology.cosmology import setCosmology, setCurrent
    dic_cosmo = _cosmo_cosmoprimo_to_colossus(cosmo)
    setCurrent(setCosmology('Geppetto', dic_cosmo))
    return concentration.concentration(M, mdef, z=z, model=cmrelation)


def process_pinocchio_box(cat, z, cosmo, cmrelation='diemer19', mdef='200c', dtype='float32'):
    from colossus.halo import mass_so

    dic = {}
    for i, vv in enumerate('xyz'):
        dic[vv] = cat['pos'].T[i].astype(dtype)
        dic['v'+vv] = cat['vel'].T[i].astype(dtype)

    dic['Mh'] = cat['Mass'].astype(dtype)
    dic['log10_Mh'] = np.log10(dic['Mh'])
    dic['halo_id']=cat['name'].astype('int64')
    dic['c']=get_concentration(dic['Mh'], z, cosmo, cmrelation).astype(dtype)
    dic['Rh'] = mass_so.M_to_R(dic['Mh'], z, mdef).astype(dtype)
    return Catalog.from_dict(dic)

def read_pinnochio_hcat(args):
    if args['hcat']['Pinnochio']['dir_sim'] is not None:
        dir_sim = args['hcat']['Pinnochio']['dir_sim']
    if socket.gethostname() == 'antoine-ThinkPad-P1-Gen-6':
        dir_sim = f'/home/antoine/Bureau/Transfert/postdoc/Euclid/data/GeppettoFC/{args["hcat"]["sim_name"]}/' 
    elif 'yggdrasil' in socket.gethostname():
        dir_sim = f'/home/users/r/rocher/scratch/Euclid/GeppettoFC/{args["hcat"]["sim_name"]}/'
    elif os.environ['NERSC_HOST'] == 'perlmutter':
        dir_sim = f'/pscratch/sd/a/arocher/Euclid/GeppettoFC/{args["hcat"]["sim_name"]}/' 
    else:
        raise NameError("Can not search for Pinnocchio simulation. Don't recognize {} ".format(socket.gethostname()))

    cat = Catalog.read(os.path.join(dir_sim, 'pinocchio.{:.4f}.{}.catalog.fits'.format(args['hcat']["z_simu"], args['hcat']["sim_name"])))
    
    if args['hcat']['mass_cut'] is not None:
        cat = cat[cat['Mass'] > args['hcat']['mass_cut']]
    
    
    input_file = open(os.path.join(dir_sim, '{}.params'.format(args['hcat']["sim_name"])), 'r')
    for ll in input_file.readlines():
        if np.in1d('BoxSize', ll.split()).any(): 
            boxsize = float(ll.split()[1])
            break  
    input_file.close()
    cosmo = Pinocchio_cosmo(args)
    '''except:
        boxsize = 1200 if 'Geppetto' in args["hcat"]["sim_name"] else 3200 if 'EuclidLargeBox' in args["hcat"]["sim_name"] else args['hcat']['boxsize']
        cosmo = Planck2015FullFlatLCDM(engine=args['cosmo']['engine'])
        import warnings
        warnings.warn('Could not find {} to read cosmological parameters and boxsize. Set boxzise to {} and cosmology to Planck2015'.format(os.path.join(dir_sim, args["hcat"]["sim_name"], '{}.params'.format(args["hcat"]["sim_name"]), boxsize)))'''
    
    
    hcat = process_pinocchio_box(cat, z=args["hcat"]["z_simu"], cosmo=cosmo, mdef=args['mass_def'], cmrelation=args['cm_relation'])
    return hcat, boxsize, cosmo


def Pinocchio_cosmo(args):
    """
    Initialize :class:`Cosmology` based on Pinnocchio .param file 

    Parameters
    ----------
    args: input argument of HOD class

    Returns
    -------
    cosmology : Cosmology
    """
    from cosmoprimo.fiducial import Cosmology
    if socket.gethostname() == 'antoine-ThinkPad-P1-Gen-6':
        dir_sim = f'/home/antoine/Bureau/Transfert/postdoc/Euclid/data/GeppettoFC/{args["hcat"]["sim_name"]}/' 

    elif 'yggdrasil' in socket.gethostname():
        dir_sim = f'/home/users/r/rocher/scratch/Euclid/GeppettoFC/{args["hcat"]["sim_name"]}/'
    elif os.environ['NERSC_HOST'] == 'perlmutter':
        dir_sim = f'/pscratch/sd/a/arocher/Euclid/GeppettoFC/{args["hcat"]["sim_name"]}/' 
    else:
        raise NameError("Can not search for Pinnocchio simulation. Don't recognize {} ".format(socket.gethostname()))
    input_file = open(os.path.join(dir_sim, f'{args["hcat"]["sim_name"]}.params'), 'r')
    keys = ['Omega0', 'OmegaLambda', 'OmegaBaryon', 'Hubble100', 'Sigma8', 'PrimordialIndex', 'DEw0', 'DEwa']
    values = []
    for ll in input_file.readlines():
        if np.in1d(keys, ll.split()).any():
            values += [float(ll.split()[1]) if 'Sigma8' not in ll.split()[0] else float(ll.split()[3])]
    input_file.close()
    cosmo_dic = dict(zip(keys,values))
    cosmo_custom = Cosmology(**dict(Omega0_m=cosmo_dic['Omega0'], Omega0_L=cosmo_dic['OmegaLambda'], Omega_b=cosmo_dic['OmegaBaryon'], h=cosmo_dic['Hubble100'], sigma8=cosmo_dic['Sigma8'], n_s=cosmo_dic['PrimordialIndex'], w0_fdl=cosmo_dic['DEw0'], wa_fdl=cosmo_dic['DEwa']), engine=args['cosmo']['engine'])
    return cosmo_custom


def _cosmo_cosmoprimo_to_colossus(cosmo):
    flat = True if cosmo.Omega0_k == 0 else False
    dic_cosmo = {'flat': flat, 
                 'H0': cosmo.H0, 
                 'Om0': cosmo.Omega0_m, 
                 'Ob0': cosmo.Omega0_b, 
                 'sigma8': cosmo.sigma8_m, 
                 'ns': cosmo.n_s, 
                 'w0':cosmo.w0_fld, 
                 'wa':cosmo.wa_fld, 
                 'Tcmb0':cosmo.T0_cmb,
                 'Neff':cosmo.N_eff}
    
    return dic_cosmo

# Now in cosmoprimo as UchuuPlanck2015
# def Planck2015FullFlatLCDM(engine=None, extra_params=None, **params):
#     """
#     Initialize :class:`Cosmology` based on Table 4 Planck2015 TT,TE,EE+lowP+lensing. 

#     Parameters
#     ----------
#     engine : string, default=None
#         Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_no
# wiggle', 'bbks'].
#         If ``None``, returns current :attr:`Cosmology.engine`.

#     extra_params : dict, default=None
#         Extra engine parameters, typically precision parameters.

#     params : dict
#         Cosmological and calculation parameters which take priority over the default ones.

#     Returns
#     -------
#     cosmology : Cosmology
#     """

#     from cosmoprimo.fiducial import Cosmology, constants

#     default_params = dict(h=0.6751, omega_cdm=0.1193, omega_b=0.02226 , Omega_k=0., sigma8=0.8150, k_pivot=0.05, n_s=0.9653, m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, tau_reio=0.063, A_L=1.0, w0_fld=-1., wa_fld=0.)
#     return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)


