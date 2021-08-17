""" Save iMPS in format of previous tenpy """

# MD, 20.07.21

import os
import pickle
import numpy as np
import bz2, glob

from mps.mps import iMPS
from models.model import create_translate_Q1


def save_mps(psi, path):
    """ Saves instance of class iMPS to several files, organized in      
        folders.

        It turned out that from a certain file size on (> 4 GB), saving the whole state in one pickled bz2-file always throws an IOError.

        psi -- instance of class iMPS
        path -- path where it should be stored
    """
    
    # Make sure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Singular values
    subpath = os.path.join(path, 'singular_values.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(psi.s, f)

    # Other parameters
    subpath = os.path.join(path, 'canonical_form.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(psi.form, f)

    subpath = os.path.join(path, 'site_pipes.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(psi.site_pipes, f)

    subpath = os.path.join(path, 'translate_Q1_data.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(psi.translate_Q1_data, f)

    # Attributes
    attributes = {
        'grouped': psi.grouped,
        'transfermatrix_keep': psi.can_keep,
        'L': psi.L,
        'max_bond_dimension': np.max(psi.chi),
        'boundary_condition': psi.bc,
    }

    subpath = os.path.join(path, 'attributes.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(attributes, f)

    # B-tensors
    path = os.path.join(path, 'tensors')
    if not os.path.exists(path):
        os.mkdir(path)

    for i, b in enumerate(psi.B):
        subpath = os.path.join(path, 'B_no{}.pkl'.format(i))
        with open(subpath, 'wb') as f:
            pickle.dump(b, f)


def load_mps(path):
    """ Load mps """

    obj = iMPS.__new__(iMPS) # create class instance, no __init__() call

    # Load singular values
    subpath = os.path.join(path, 'singular_values.pkl')
    with open(subpath, 'rb') as f:
        obj.s = pickle.load(f)

    # Attributes
    subpath = os.path.join(path, 'attributes.pkl')
    with open(subpath, 'rb') as f:
        attributes = pickle.load(f)

    obj.bc = attributes['boundary_condition']
    obj.L = attributes['L']

    # Get B-tensors
    path_t = os.path.join(path, 'tensors')
    Bs = []
    for i in np.arange(obj.L):
        subpath = os.path.join(path_t, 'B_no{}.pkl'.format(i))
        with open(subpath, 'rb') as f:
            Bs.append(pickle.load(f))

    obj.B = Bs

    # Load other parameters
    subpath = os.path.join(path, 'canonical_form.pkl')
    with open(subpath, 'rb') as f:
        obj.form = pickle.load(f)

    subpath = os.path.join(path, 'site_pipes.pkl')
    with open(subpath, 'rb') as f:
        obj.site_pipes = pickle.load(f)


    obj.init_from_B()


    obj.grouped = attributes['grouped']
    obj.can_keep = attributes['transfermatrix_keep']

    subpath = os.path.join(path, 'translate_Q1_data.pkl')
    with open(subpath, 'rb') as f:
        obj.translate_Q1_data = pickle.load(f)
    
    obj.translate_Q1 = create_translate_Q1(obj.translate_Q1_data)

    obj.check_sanity()
    return obj


def save_LP(LP, path):
    """ Save LP """
    if not os.path.exists(path):
        os.mkdir(path)

    for i, lp in enumerate(LP):
        subpath = os.path.join(path, 'LP_no{}.pkl'.format(i))
        with open(subpath, 'wb') as f:
            pickle.dump(lp, f)


def save_RP(RP, path):
    """ Save RP """
    if not os.path.exists(path):
        os.mkdir(path)
        
    for i, rp in enumerate(RP):
        subpath = os.path.join(path, 'RP_no{}.pkl'.format(i))
        with open(subpath, 'wb') as f:
            pickle.dump(rp, f)


def load_LP(path):
    """ Load LP """
    # Find number of files
    subpath = os.path.join(path, 'LP_no*.pkl')
    files = sorted(glob.glob(subpath), key=os.path.getmtime)
    L = len(files)
    LP = []
    for i in np.arange(L):
        subpath = os.path.join(path, 'LP_no{}.pkl'.format(i))
        with open(subpath, 'rb') as f:
            LP.append(pickle.load(f))
    return LP
        

def load_RP(path):
    """ Load RP """
    # Find number of files
    subpath = os.path.join(path, 'RP_no*.pkl')
    files = sorted(glob.glob(subpath), key=os.path.getmtime)
    L = len(files)
    RP = []
    for i in np.arange(L):
        subpath = os.path.join(path, 'RP_no{}.pkl'.format(i))
        with open(subpath, 'rb') as f:
            RP.append(pickle.load(f))
    return RP


    
def save_env(env, path):
    """ Saves instance of class environment1 to several files, organized in folders.
    
    It turned out that from a certain file size on (> 4 GB), saving the whole environment in one pickled bz2-file always throws an IOError (17.08.2021).

    In python3, this can be resolved by using pickle with a higher protocol.

    env -- instance of class environment1
    path -- path where to store it
    """

    # Make sure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # psi
    subpath = os.path.join(path, 'psi')
    save_mps(env.ket, subpath)

    # LP
    subpath = os.path.join(path, 'LP')
    save_LP(env.LP, subpath)

    # RP
    subpath = os.path.join(path, 'RP')
    save_RP(env.RP, subpath)

    # W
    subpath = os.path.join(path, 'mpo.pkl')
    with open(subpath, 'wb') as f:
        pickle.dump(env.W, f)


def load_env(path):
    """ Load env """
    
    # psi
    subpath = os.path.join(path, 'psi')
    psi = load_mps(subpath)

    # LP
    subpath = os.path.join(path, 'LP')
    LP = load_LP(subpath)

    # RP
    subpath = os.path.join(path, 'RP')
    RP = load_RP(subpath)

    # W
    subpath = os.path.join(path, 'mpo.pkl')
    with open(subpath, 'rb') as f:
        W = pickle.load(f)

    from mps.enviro import make_env1_from_LPRP
    env = make_env1_from_LPRP(LP, RP, W, psi, override_labels=True)

    return env


    
