""" Get size of instance of iMPS in previous tenpy

"""
# MD, 19.07.2021

import numpy as np
import sys



def get_size_npc_array(B):
    """ 

        Format: previous tenpy
    """
    list_keys = B.__dict__.keys()
    total_size = 0.

    for k in list_keys:
        total_size += sys.getsizeof(B.__dict__[k])

    # Consider B.dat
    for d in B.dat:
        total_size += sys.getsizeof(d)

    for d in B.q_ind:
        total_size += sys.getsizeof(d)

    return total_size


def get_size_mps(psi):
    """ Returns the size of the mps 

        Format: previous tenpy
    """
    list_keys = psi.__dict__.keys()
    total_size = 0.

    for k in list_keys:
        total_size += sys.getsizeof(psi.__dict__[k])

    # Consider B-tensors separately
    for b in psi.B:
        total_size += get_size_npc_array(b)

    return total_size