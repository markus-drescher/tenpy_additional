""" J1J2-Heisenberg model on the triangular lattice
"""

import numpy as np

from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingMPOModel


class heisenberg_triangular(CouplingMPOModel):
    r"""Spin-1/2 sites coupled by next-nearest neighbour interactions

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i, j \rangle, i < j}
               2 \cdot \mathtt{J^{(1)}_{xy}} [S^+_i S^-_j + S^-_i S^+_j]
               + 4 \cdot mathtt{J^{(1)}_{z}} S^z_i S^z_j
        +   \sum_{\llangle i, j \rrangle, i < j}
               2 \cdot \mathtt{J^{(2)}_{xy}} [S^+_i S^-_j + S^-_i S^+_j]
               + 4 \cdot mathtt{J^{(2)}_{z}} S^z_i S^z_j

    """
    def init_sites(self, model_params):
        S = 0.5
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz'

        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        # read out/set default parameters
        J1xy = model_params.get('J1xy', 1.)
        J1z = model_params.get('J1z', 1.)
        J2xy = model_params.get('J2xy', 0.125)
        J2z = model_params.get('J2z', 0.125)

        print('*'*80)
        print('Nearest neighbour pairs')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(2*J1xy, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(4*J1z, u1, 'Sz', u2, 'Sz', dx, plus_hc=False)
            print('u1, u2, dx', u1, u2, dx)

        print('*'*80)
        print('Next-nearest neighbour pairs')
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(2*J2xy, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(4*J2z, u1, 'Sz', u2, 'Sz', dx, plus_hc=False)
            print('u1, u2, dx', u1, u2, dx)


class heisenberg_triangular_flux(CouplingMPOModel):
    r"""Spin-1/2 sites coupled by next-nearest neighbour interactions

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i, j \rangle, i < j}
               2 \cdot \mathtt{J^{(1)}_{xy}} [S^+_i S^-_j + S^-_i S^+_j]
               + 4 \cdot mathtt{J^{(1)}_{z}} S^z_i S^z_j
        +   \sum_{\llangle i, j \rrangle, i < j}
               2 \cdot \mathtt{J^{(2)}_{xy}} [S^+_i S^-_j + S^-_i S^+_j]
               + 4 \cdot mathtt{J^{(2)}_{z}} S^z_i S^z_j

    """
    def init_sites(self, model_params):
        S = 0.5
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz'

        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        # read out/set default parameters
        J1xy = model_params.get('J1xy', 1.)
        J1z = model_params.get('J1z', 1.)
        J2xy = model_params.get('J2xy', 0.125)
        J2z = model_params.get('J2z', 0.125)
        theta = model_params.get('theta', 0.)

        print('*'*80)
        print('Nearest neighbour pairs')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            strength_with_flux = self.coupling_strength_add_ext_flux(2*J1xy, dx, [0, theta])
            self.add_coupling(strength_with_flux, u1, 'Sp', u2, 'Sm', dx, plus_hc=False)
            self.add_coupling(np.conj(strength_with_flux), u2, 'Sp', u1, 'Sm', -dx, plus_hc=False)
            self.add_coupling(4*J1z, u1, 'Sz', u2, 'Sz', dx, plus_hc=False)
            print('u1, u2, dx', u1, u2, dx)

        print('*'*80)
        print('Next-nearest neighbour pairs')
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            strength_with_flux = self.coupling_strength_add_ext_flux(2*J2xy, dx, [0, theta])
            self.add_coupling(strength_with_flux, u1, 'Sp', u2, 'Sm', dx, plus_hc=False)
            self.add_coupling(np.conj(strength_with_flux), u2, 'Sp', u1, 'Sm', -dx, plus_hc=False)
            self.add_coupling(4*J2z, u1, 'Sz', u2, 'Sz', dx, plus_hc=False)
            print('u1, u2, dx', u1, u2, dx)


class heisenberg_triangular_NN(CouplingMPOModel):
    r"""Spin-1/2 sites coupled by next-nearest neighbour interactions

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i, j \rangle, i < j}
               2 \cdot \mathtt{J^{(1)}_{xy}} [S^+_i S^-_j + S^-_i S^+_j]
               + 4 \cdot mathtt{J^{(1)}_{z}} S^z_i S^z_j

    """
    def init_sites(self, model_params):
        S = 0.5
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz'

        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        # read out/set default parameters
        J1xy = model_params.get('J1xy', 1.)
        J1z = model_params.get('J1z', 1.)

        print('*'*80)
        print('Nearest neighbour pairs')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(2*J1xy, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(4*J1z, u1, 'Sz', u2, 'Sz', dx, plus_hc=False)
            print('u1, u2, dx', u1, u2, dx)
