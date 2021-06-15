""" Transverse field Ising model on a square lattice with standard order
    of states (up, down). Uses SpinHalfSite.

    Written for usage with MPS for ising_square model in the old tenpy, which
    have been converted to the new tenpy format.
"""

import numpy as np

from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import asConfig


class Ising_square(CouplingMPOModel):
    r"""Spin-1/2 sites coupled by next-nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = -\sum_{\langle i, j \rangle, i < j}
                \mathtt{J} \sigma^x_i \sigma^x_j 
            - \sum_i
                \mathtt{g} \sigma^z_i

    """
    def init_sites(self, model_params):
        S = 0.5
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'parity'
        
        site = SpinHalfSite(conserve)
        return site

    def init_terms(self, model_params):
        # read out/set default parameters
        J = model_params.get('J', 1.)
        g = model_params.get('g', 1.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g*2, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J*4, u1, 'Sx', u2, 'Sx', dx, plus_hc=False)
        

class Ising_square2(CouplingMPOModel):
    r"""Spin-1/2 sites coupled by next-nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = -\sum_{\langle i, j \rangle, i < j}
                \mathtt{J} \sigma^x_i \sigma^x_j 
            - \sum_i
                \mathtt{g} \sigma^z_i

    Note: This model is physically equivalent to Ising_square.
    The implementation is adapted from the predefined model
    SpinChainNNN2. However, the virtual bond dimension is 
    larger than for Ising_square and hence less efficient.

    """
    def init_sites(self, model_params):
        S = 0.5
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'parity'
        
        site = SpinHalfSite(conserve)
        return site

    def init_terms(self, model_params):
        # read out/set default parameters
        J = model_params.get('J', 1.)
        g = model_params.get('g', 1.)
        Jx = J
        Jy = 0.

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g*2, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-(Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(-(Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            
