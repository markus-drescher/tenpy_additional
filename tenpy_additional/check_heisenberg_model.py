import numpy as np

from tenpy_additional.heisenberg_triangular_model import heisenberg_triangular
from tenpy_additional.heisenberg_triangular_model import heisenberg_triangular_NN

from functions_heisenberg.check_and_print_model import print_mpo_matrix

L = 6
Lx = 2
J1xy = 1.
J1z = 1.
J2xy = 0.125
J2z = 0.125
bc = 'infinite'
conserve = 'Sz'


model_params = dict(Lx=Lx, Ly=L,
                    J1xy = J1xy, J1z = J1z,
                    J2xy = J2xy, J2z = J2z,
                    bc_MPS = bc, conserve = conserve,
                    lattice = 'Triangular',
                    bc_y = 'cylinder',
                    )

M = heisenberg_triangular(model_params)
print(M.H_MPO.chi)

model_params2 = dict(Lx=Lx, Ly=L,
                    J1xy = J1xy, J1z = J1z,
                    bc_MPS = bc, conserve = conserve,
                    lattice = 'Triangular',
                    bc_y = 'cylinder',
                    )

M2 = heisenberg_triangular_NN(model_params2)
print(M2.H_MPO.chi)

print('site 0')
print_mpo_matrix(M2.H_MPO.get_W(0).to_ndarray())
print('\nsite 1')
print_mpo_matrix(M2.H_MPO.get_W(1).to_ndarray())
print('\nsite 2')
print_mpo_matrix(M2.H_MPO.get_W(2).to_ndarray())