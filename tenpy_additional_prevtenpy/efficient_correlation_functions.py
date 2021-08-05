""" Calculate correlation function efficienctly """

# MD, July 29th, 2019

import numpy as np
from mps.mps import iMPS

def corr_tx_array(psi, phi0, op, site_start, site_final):
    """ psi -- time evolved state (the ket state)
        phi0 -- ground state (the bra state)
        op -- operator (or list of operators) to be applied (npc-operators)
        op = [op[0], op[1], ...]    or   op = op[0]
        =================================================================
        NOTA BENE: The operators in op are applied to the ket state, i.e. 
        <phi0| op | psi> .                                              
        =================================================================
        site_start -- first site where to apply op
        site_final -- last site where to apply op
    """

    R = []
    L = []

    N = psi.L
    if phi0.L != N:
        print("Error: psi.L != phi0.L")
        return

    if not(isinstance(op, list)):
        op = [op] * (site_final-site_start+1)

    if not len(op)==(site_final-site_start+1):
        print("Error: List of operators has wrong length!")

    # Calculate right environments up to site_start+1 and store them on the fly

    R_0 = np.ones((1,1))
    R.append(R_0)

    for i in np.arange(N-1, site_start, -1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        R_i = np.tensordot(B_bra, 
                np.tensordot(B_ket, R[0], axes=(2,1)), 
                axes=([0,2],[0,2]),
                )
        R.insert(0, R_i)

    # Build left environments
    L_0 = np.ones((1,1))
    L.append(L_0)
    
    for i in np.arange(0, site_start):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)), 
                B_ket, 
                axes=([0,1],[1,0]),
                )

        L.append(L_i)

    Corr = []
    
    # iterate over all sites where op is applied
    for i in np.arange(site_start, site_final+1):
        # Contract matrices with the operator
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        op_i = op[i-site_start].to_ndarray()
        temp = np.tensordot(op_i, B_ket, axes=(1,0))
        Bop = np.tensordot(B_bra,
                    temp,
                    axes=(0,0),
                    )
        # Contract with the environments
        res = np.tensordot(L[-1],
                    np.tensordot(Bop, R[0], axes=([1,3],[0,1])),
                    axes=([0,1],[0,1]),
                    )
        # Update the stored environments
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)),
                B_ket,
                axes=([0,1],[1,0]),
                )
        L.append(L_i)
        
        del R[0]
        
        # Save result
        Corr.append(res)

    return Corr



def corr_tx_array_periodic(psi, phi0, op, site_start, site_final, i_exc):
    """ psi -- time evolved state (the ket state)
        phi0 -- ground state (the bra state)
        op -- operator (or list of operators) to be applied (npc-operators)
        op = [op[0], op[1], ...]    or   op = op[0]
        =================================================================
        NOTA BENE: The operators in op are applied to the ket state, i.e. 
        <phi0| op | psi> .                                              
        =================================================================
        site_start -- first site where to apply op
        site_final -- last site where to apply op
        i_exc -- (site) position of the excitation
        =================================================================
        For periodic boundary conditions
    """

    R = []
    L = []

    N = psi.L
    if phi0.L != N:
        print("Error: psi.L != phi0.L")
        return

    if not(isinstance(op, list)):
        op = [op] * (site_final-site_start+1)

    if not len(op)==(site_final-site_start+1):
        print("Error: List of operators has wrong length!")


    # Calculate right environments up to site_start+1 and store them on the fly

    # First, initialize the right and left vectors (kind of power method)
    Chi_psi = psi.chi[-1]
    Chi_phi0 = phi0.chi[-1]

    vr = np.random.rand(Chi_psi * Chi_phi0)
    vr = vr/np.linalg.norm(vr)
    vr = vr.reshape(Chi_phi0, Chi_psi)
    

    vl = np.random.rand(Chi_psi*Chi_phi0)
    vl = vl/np.linalg.norm(vl)
    vl = vl.reshape(Chi_phi0, Chi_psi)


    # Build right environments
    R.append(vr)

    for i in np.arange(N-1, site_start, -1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        R_i = np.tensordot(B_bra, 
                np.tensordot(B_ket, R[0], axes=(2,1)),
                axes=([0,2],[0,2]),
                )
        R.insert(0, R_i)

    # Build left environments
    L.append(vl/np.sum(vl*vr))

    for i in np.arange(0, site_start):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)), 
                B_ket, 
                axes=([0,1],[1,0]),
                )

        L.append(L_i)

    Corr = []
    
    # iterate over all sites where op is applied
    for i in np.arange(site_start, site_final+1):
        # Contract matrices with the operator
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        op_i = op[i-site_start].to_ndarray()
        temp = np.tensordot(op_i, B_ket, axes=(1,0))
        Bop = np.tensordot(B_bra,
                    temp,
                    axes=(0,0),
                    )
        # Contract with the environments
        res = np.tensordot(L[-1],
                    np.tensordot(Bop, R[0], axes=([1,3],[0,1])),
                    axes=([0,1],[0,1]),
                    )
        # Update the stored environments
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)),
                B_ket,
                axes=([0,1],[1,0]),
                )
        L.append(L_i)
        
        del R[0]
        
        # Save result
        Corr.append(res)

    return Corr



def corr_tx_array_periodic_Ruben(psi, phi0, op, site_start, site_final, i_exc):
    """ psi -- time evolved state (the ket state)
        phi0 -- ground state (the bra state)
        op -- operator (or list of operators) to be applied (npc-operators)
        op = [op[0], op[1], ...]    or   op = op[0]
        =================================================================
        NOTA BENE: The operators in op are applied to the ket state, i.e. 
        <phi0| op | psi> .                                              
        =================================================================
        site_start -- first site where to apply op
        site_final -- last site where to apply op
        i_exc -- (site) position of the excitation
        =================================================================
        For periodic boundary conditions
        Original idea adopted from Ruben Verresen.
    """

    R = []
    L = []

    N = psi.L
    if phi0.L != N:
        print("Error: psi.L != phi0.L")
        return

    if not(isinstance(op, list)):
        op = [op] * (site_final-site_start+1)

    if not len(op)==(site_final-site_start+1):
        print("Error: List of operators has wrong length!")


    # Calculate right environments up to site_start+1 and store them on the fly

    # First, initialize the right and left vectors (kind of power method)
    Chi_psi = psi.chi[i_exc-1]
    Chi_phi0 = phi0.chi[i_exc-1]

    vr = np.random.rand(Chi_psi * Chi_phi0)
    vr = vr/np.linalg.norm(vr)
    vr = vr.reshape(Chi_phi0, Chi_psi)
    for i in np.arange(i_exc-1, -1, -1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        vr = np.tensordot(B_bra, 
                    np.tensordot(B_ket, vr, axes=(2,1)), 
                    axes=([0,2],[0,2]),
                    )

    Chi_psi = psi.chi[i_exc]
    Chi_phi0 = phi0.chi[i_exc]

    vl = np.random.rand(Chi_psi*Chi_phi0)
    vl = vl/np.linalg.norm(vl)
    vl = vl.reshape(Chi_phi0, Chi_psi)
    for i in np.arange(i_exc+1, psi.L, 1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        vl = np.tensordot(
                np.tensordot(vl, B_bra, axes=(0,1)), 
                B_ket, 
                axes=([0,1],[1,0]),
                )


    R.append(vr)

    for i in np.arange(N-1, site_start, -1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        R_i = np.tensordot(B_bra, 
                np.tensordot(B_ket, R[0], axes=(2,1)), 
                axes=([0,2],[0,2]),
                )
        R.insert(0, R_i)

    # Build left environments
    L.append(vl/np.sum(vl*vr))

    for i in np.arange(0, site_start):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)), 
                B_ket, 
                axes=([0,1],[1,0]),
                )

        L.append(L_i)

    Corr = []
    
    # iterate over all sites where op is applied
    for i in np.arange(site_start, site_final+1):
        # Contract matrices with the operator
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi0.B[i].to_ndarray().conj()
        op_i = op[i-site_start].to_ndarray()
        temp = np.tensordot(op_i, B_ket, axes=(1,0))
        Bop = np.tensordot(B_bra,
                    temp,
                    axes=(0,0),
                    )
        # Contract with the environments
        res = np.tensordot(L[-1],
                    np.tensordot(Bop, R[0], axes=([1,3],[0,1])),
                    axes=([0,1],[0,1]),
                    )
        # Update the stored environments
        L_i = np.tensordot(
                np.tensordot(L[-1], B_bra, axes=(0,1)),
                B_ket,
                axes=([0,1],[1,0]),
                )
        L.append(L_i)
        
        del R[0]
        
        # Save result
        Corr.append(res)

    return Corr



def calculate_corr_faster(psi, phi, Op, site_start, site_final, i_exc):
    # psi: time-evolved state
    # phi: gs
    # Op: operator of the excitation
    # index_list: list of of the positions where to apply operator
    # i_exc: position of the excitation (time t)
    index_list = [i for i in np.arange(site_start, site_final+1)]
    chi0 = psi.chi[i_exc-1]; chi1 = phi.chi[i_exc-1]
    vr = np.random.rand(chi0*chi1)
    vr = vr/np.linalg.norm(vr)
    vr = vr.reshape(chi0,chi1)
    for i in np.arange(i_exc-1,-1,-1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        vr = np.tensordot(np.tensordot(B_ket,vr,axes=(2,0)),B_bra,axes=([0,2],[0,2]))
    #print 'vr = ', np.linalg.norm(vr)
    chi0 = psi.chi[i_exc]; chi1 = phi.chi[i_exc]
    vl = np.random.rand(chi0*chi1)
    vl = vl/np.linalg.norm(vl)
    vl = vl.reshape(chi0,chi1)
    for i in np.arange(i_exc+1,psi.L,1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        vl = np.tensordot(np.tensordot(vl,B_ket,axes=(0,1)),B_bra,axes=([0,1],[1,0]))
        #print 'vl_temp = ', np.linalg.norm(vl)
    #print 'vl = ', np.linalg.norm(vl)

    R = []
    L = []
    R.append(vr)
    L.append(vl/np.sum(vl*vr))
    for i in range(psi.L)[::-1]:
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        v_r = R[-1]
        R.append(np.tensordot(np.tensordot(B_ket,v_r,axes=(2,0)),B_bra,axes=([0,2],[0,2])))
    for i in range(psi.L):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        v_l = L[-1]
        L.append(np.tensordot(np.tensordot(v_l,B_ket,axes=(0,1)),B_bra,axes=([0,1],[1,0])))
    
    corr = []
    
    Op = Op.to_ndarray()

    for index in index_list:
        B_ket = np.tensordot(Op,psi.B[index].to_ndarray(),axes=(1,0))
        B_bra = phi.B[index].to_ndarray().conj()
        index = np.mod(index,psi.L)
        temp = np.tensordot(np.tensordot(B_ket,R[-(index+2)],axes=(2,0)),B_bra,axes=([0,2],[0,2]))
        temp = np.tensordot(L[index],temp,axes=([0,1],[0,1]))
        if np.abs(temp) < 10**(-13): temp = 0
        else: temp = temp #/np.tensordot(L[0],R[-1],axes=([0,1],[0,1]))
        corr.append(temp)
    return corr


def calculate_corr(psi, phi, Op_list, site_start, site_final, i_exc):
    # psi: time-evolved state
    # phi: gs
    # Op_list: list operators of the excitation
    # i_exc: position of the excitation (time t)

    index_list = [i for i in np.arange(site_start, site_final+1)]
    chi0 = psi.chi[i_exc-1]; chi1 = phi.chi[i_exc-1]
    vr = np.random.rand(chi0*chi1)
    vr = vr/np.linalg.norm(vr)
    vr = vr.reshape(chi0,chi1)
    for i in np.arange(i_exc-1,-1,-1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        vr = np.tensordot(np.tensordot(B_ket,vr,axes=(2,0)),B_bra,axes=([0,2],[0,2]))
    #print 'vr = ', np.linalg.norm(vr)
    chi0 = psi.chi[i_exc]; chi1 = phi.chi[i_exc]
    vl = np.random.rand(chi0*chi1)
    vl = vl/np.linalg.norm(vl)
    vl = vl.reshape(chi0,chi1)
    for i in np.arange(i_exc+1,psi.L,1):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        vl = np.tensordot(np.tensordot(vl,B_ket,axes=(0,1)),B_bra,axes=([0,1],[1,0]))
        #print 'vl_temp = ', np.linalg.norm(vl)
    #print 'vl = ', np.linalg.norm(vl)

    R = []
    L = []
    R.append(vr)
    L.append(vl/np.sum(vl*vr))
    for i in range(psi.L)[::-1]:
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        v_r = R[-1]
        R.append(np.tensordot(np.tensordot(B_ket,v_r,axes=(2,0)),B_bra,axes=([0,2],[0,2])))
    for i in range(psi.L):
        B_ket = psi.B[i].to_ndarray()
        B_bra = phi.B[i].to_ndarray().conj()
        v_l = L[-1]
        L.append(np.tensordot(np.tensordot(v_l,B_ket,axes=(0,1)),B_bra,axes=([0,1],[1,0])))
    
    corr = []

    for index in index_list:
        Op = Op_list[index-site_start].to_ndarray()
        B_ket = np.tensordot(Op,psi.B[index].to_ndarray(),axes=(1,0))
        B_bra = phi.B[index].to_ndarray().conj()
        index = np.mod(index,psi.L)
        temp = np.tensordot(np.tensordot(B_ket,R[-(index+2)],axes=(2,0)),B_bra,axes=([0,2],[0,2]))
        temp = np.tensordot(L[index],temp,axes=([0,1],[0,1]))
        if np.abs(temp) < 10**(-13): temp = 0
        else: temp = temp #/np.tensordot(L[0],R[-1],axes=([0,1],[0,1]))
        corr.append(temp)
    return corr
