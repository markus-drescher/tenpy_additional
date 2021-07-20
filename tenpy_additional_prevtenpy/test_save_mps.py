import numpy as np
from mps.mps import iMPS
from models import ising_square
from tenpy_additional_prevtenpy.save_mps import save_mps, load_mps
import os

L = 4
mult = 3
g = 0.2
J = 1
cp = True

model_par = {
    'L': L,
    'g': g,
    'J': J,
    'conserve parity': cp,
}

M = ising_square.ising_square(model_par)

init_state = np.array([0, 1] * ((L*mult)//2))
psi_0 = iMPS.product_imps(
    d = M.d,
    p_state = init_state,
    dtype=complex,
    bc = 'periodic',
    conserve = M,
)

fileDir = os.path.dirname(os.path.realpath('__file__'))
pathname = os.path.join(fileDir, 'psi_0')
save_mps(psi_0, pathname)


load_mps(pathname)