# MD, 15.02.2021

import numpy as np
# import pylab as pl
# import matplotlib.pyplot as plt
from tenpy.version import version_summary
from tenpy.networks.mps import MPS, MPSEnvironment
from tenpy.networks.mpo import MPOEnvironment
from tenpy.models.spins_nnn import SpinChainNNN2
from tenpy.models.model import NearestNeighborModel
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
from tenpy.linalg import np_conserved as npc
from tenpy.tools import hdf5_io
import time
import h5py


import bz2
import pickle

from .ising_square_model_tenpy import Ising_square


def chi_list(chi_min, chi_max, dchi=20, nsweeps=10):
    """ chi_list for dmrg_params """
    chi_max = int(chi_max)
    nsweeps = int(nsweeps)
    dchi = int(dchi)
    if chi_min > chi_max:
        chi_tmp = chi_min
        chi_min = chi_max
        chi_max = chi_tmp
    if chi_max < dchi:
        return {0: chi_max}
    chi_min = chi_max - (chi_max-chi_min)//dchi * dchi
    chi_min = int(chi_min)
    if chi_min==chi_max:
        return {0: chi_max}

    chi_list = {}
    for i in range((chi_max-chi_min)//dchi + 1):
        chi = chi_min + i*dchi
        chi_list[nsweeps * i] = chi
    return chi_list



def energy_finite_MPS(psi, M):
    """ Computes the energy of an MPS for a model M by contracting the full finite network.

        Written for the new tenpy and python3.
    """
    # Hamiltonian written as an MPO
    H_MPO = M.H_MPO
    env = MPOEnvironment(psi, H_MPO, psi)
    E = env.full_contraction(i0 = psi.L//2)
    assert(np.imag(E) < 1e-10)
    return np.real(E)
    

def get_norm(psi):
    """ New tenpy """
    env = MPSEnvironment(psi, psi)
    norm = env.full_contraction(i0=psi.L//2)
    assert(np.imag(norm) < 1e-10)
    return np.sqrt(np.real(norm))


def print_B_new(B_new):
    """ new tenpy """

    print('-'*80)
    print('labels ', B_new._labels)
    print('legs ', B_new.legs)
    print('chinfo ', B_new.chinfo)
    print('_qdata ', B_new._qdata)
    print('_data ', B_new._data)
    print('-'*80)


if __name__ == "__main__":

    Lx = 8
    Ly = 4
    g = 2.498
    chi_max = 200
    chi_start_dmrg = 10
    dchi_dmrg = 10
    nsweeps = 10

    # Predefined model
    model_params = dict(Lx=Lx, Ly=Ly,
                            Jx=-4., Jy=0., Jz=0.,
                            Jxp=0., Jyp=0., Jzp=0.,
                            hz=2.*g,
                            bc_MPS='finite', conserve='parity', verbose=2,
                            lattice = 'Square',
                            bc_y = 'cylinder')

    M_GS = SpinChainNNN2(model_params)


    # My own model
    model_params = dict(Lx=Lx, Ly=Ly,
                        J = 1., g = 2.498,
                        bc_MPS='finite', conserve='parity', verbose=2,
                        lattice = 'Square',
                        bc_y = 'cylinder')

    myM = Ising_square(model_params)


    # Load ground state found from DMRG in the new tenpy
    name = 'tenpy_ising_square_GS_MPS_Ly{}_Lx{}_g{}_chi_max{}_cpTrue_finite.out'.format(Ly, Lx, g, chi_max)

    with bz2.BZ2File(name, 'rb') as out_obj:
        psi = pickle.load(out_obj)

    print('-'*80)
    print('Ground state found from DMRG in the new tenpy.')
    print('physical leg ', psi.get_B(0).legs[1])
    print('qtotal ', psi.get_B(0).qtotal)

    print('Energy ', energy_finite_MPS(psi, M_GS))
    print('norm ', get_norm(psi))
    print('-'*80)


    # Load converted ground state
    name = 'tenpy_ising_square_GS_MPS_Ly{}_Lx{}_g{}_chi_max{}_cpTrue_finite_from_hdf5.out'.format(Ly, Lx, g, chi_max)

    with bz2.BZ2File(name, 'rb') as out_obj:
        psi2 = pickle.load(out_obj)

    print('Converted Ground state (from old tenpy)')
    print('physical leg ', psi2.get_B(0).legs[1])
    print('qtotal ', psi2.get_B(0).qtotal)
    print('Energy ', energy_finite_MPS(psi2, myM))
    print('norm ', get_norm(psi2))
    print('-'*80)

    print('Note that we need to use a different model (with the correct ordering of states).')
