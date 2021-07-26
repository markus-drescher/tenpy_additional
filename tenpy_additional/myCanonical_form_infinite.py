""" Modified version of the MPS class methods _canonical_form_infinite() 
    and apply_local_op
"""
# MD, 21st April, 2020

import numpy as np
import warnings
import random
from functools import reduce
import scipy.sparse.linalg.eigen.arpack

from tenpy.linalg import np_conserved as npc
from tenpy.linalg import sparse
from tenpy.networks.site import GroupedSite, group_sites
from tenpy.tools.misc import to_iterable, argsort
from tenpy.tools.math import lcm, speigs, entropy
from tenpy.algorithms.truncation import TruncationError, svd_theta


def myApply_local_op(mps, i, op, unitary=None, renormalize=False, truncate=False, chi_list=None, cutoff=1.e-13):
        """Apply a local (one or multi-site) operator to mps (Instance of class MPS).
        Returns mps.

        Note that this destroys the canonical form if the local operator is non-unitary.
        Therefore, this function calls :meth:`canonical_form` if necessary.

        Parameters
        ----------
        i : int
            (Left-most) index of the site(s) on which the operator should act.
        op : str | npc.Array
            A physical operator acting on site `i`, with legs ``'p', 'p*'`` for a single-site
            operator or with legs ``['p0', 'p1', ...], ['p0*', 'p1*', ...]`` for an operator
            acting on `n`>=2 sites.
            Strings (like ``'Id', 'Sz'``) are translated into single-site operators defined by
            :attr:`sites`.
        unitary : None | bool
            Whether `op` is unitary, i.e., whether the canonical form is preserved (``True``)
            or whether we should call :meth:`canonical_form` (``False``).
            ``None`` checks whether ``norm(op dagger(op) - identity)`` is smaller than `cutoff`.
        renormalize : bool
            Whether the final state should keep track of the norm (False, default) or be
            renormalized to have norm 1 (True).
        cutoff : float
            Cutoff for singular values if `op` acts on more than one site (see :meth:`from_full`).
            (And used as cutoff for a unspecified `unitary`.)
        truncate : bool; if true, chi_list is forwarded to myCanonical_form_infinite and the 
            bonds are truncated accordingly after applying the operator.
        chi_list: list of integers; if None and truncate == True, chi_list is set to mps.chi
        """
        if mps.bc != 'infinite':
            raise ValueError('Not implemented: Please use apply_local_op of the class MPS.')

        if truncate and chi_list == None:
            chi_list = mps.chi.copy()

        i = mps._to_valid_index(i)
        if isinstance(op, str):
            op = mps.sites[i].get_op(op)
        n = op.rank // 2  # same as int(rank/2)
        if n == 1:
            pstar, p = 'p*', 'p'
        else:
            p = mps._get_p_labels(n, False)
            pstar = mps._get_p_labels(n, True)
        if unitary is None:
            op_op_dagger = npc.tensordot(op, op.conj(), axes=[pstar, p])
            if n > 1:
                op_op_dagger = op_op_dagger.combine_legs([p, pstar], qconj=[+1, -1])
            unitary = npc.norm(op_op_dagger - npc.eye_like(op_op_dagger)) < cutoff
        if n == 1:
            opB = npc.tensordot(op, mps._B[i], axes=['p*', 'p'])
            mps.set_B(i, opB, mps.form[i])
        else:
            th = mps.get_theta(i, n)
            th = npc.tensordot(op, th, axes=[pstar, p])
            # use MPS.from_full to split the sites
            split_th = mps.from_full(mps.sites[i:i + n], th, None, cutoff, False, 'segment',
                                      (mps.get_SL(i), mps.get_SR(i + n - 1)))
            for j in range(n):
                mps.set_B(i + j, split_th._B[j], split_th.form[j])
            for j in range(n - 1):
                mps.set_SR(i + j, split_th._S[j + 1])
        if not unitary:
            mps = myCanonical_form_infinite(mps, renormalize, chi_list)
        return mps


def myCanonical_form_infinite(mps, renormalize=True, chi_list=None, tol_xi=1.e6):
        """Bring an infinite MPS into canonical form (in place).

        If any site is in :attr:`form` ``None``, it does *not* use any of the singular values `S`.
        If all sites have a `form`, it respects the `form` to ensure
        that one `S` is included per bond.
        The final state is always in right-canonical 'B' form.

        Proceeds in three steps, namely 1) diagonalize right and left transfermatrix on a given
        bond to bring that bond into canonical form, and then
        2) sweep right to left, and 3) left to right to bringing other bonds into canonical form.

        .. warning :
            You might *loose* precision when calling this function.
            When we diagonalize the transfermatrix, we get the singular values squared as
            eigenvalues, with numerical noise on the order of machine precision (usually ~1.e-15).
            Taking the square root, the new singular values are only precise to *half* the machine
            precision (usually ~1.e-7).

        Parameters
        ----------
        renormalize: bool
            Whether a change in the norm should be discarded or used to update :attr:`norm`.
        tol_xi : float
            Raise an error if the correlation length is larger than that
            (which indicates a degenerate "cat" state, e.g., for spontaneous symmetry breaking).
        """
        assert not mps.finite
        if chi_list == None:
            chi_list = mps.chi
        
        i1 = np.argmin(mps.chi)  # start at this bond
        if any([(f is None) for f in mps.form]):
            # ignore any 'S' and canonical form, just state that we are in 'B' form
            mps.form = mps._parse_form('B')
            mps._S[i1] = np.ones(mps.chi[i1], dtype=np.float)  # (is later used for guess of Gl)
        else:
            # was in canonical form before; bring back into canonical form
            # -> make sure we don't use multiple S on one bond in our definition of the MPS
            mps.convert_form('B')
        L = mps.L
        Wr_list = [None] * L  # right eigenvectors of TM on each bond after ..._correct_right

        # phase 1: bring bond (i1-1, i1) in canonical form
        # find dominant right eigenvector
        norm, Gr = mps._canonical_form_dominant_gram_matrix(i1, False, tol_xi)
        mps._B[i1] /= np.sqrt(norm)  # correct norm
        if not renormalize:
            mps.norm *= np.sqrt(norm)
        # make Gr diagonal to Wr
        Wr, Kl, Kr = mps._canonical_form_correct_right(i1, Gr)
        # guess for Gl
        Gl = npc.tensordot(Kr.scale_axis(mps.get_SL(i1)**2, 1), Kl, axes=['vR', 'vL'])
        Gl.iset_leg_labels(['vR*', 'vR'])
        # find dominant left eigenvector
        norm, Gl = mps._canonical_form_dominant_gram_matrix(i1, True, tol_xi, Gl)
        # norm = dominant left eigenvalue, Gl = dominant left eigenvector
        if abs(1. - norm) > 1.e-13:
            mps._B[i1] /= np.sqrt(norm) # correct norm again
            print('This is myCanonical_form_infinite:')
            print('The dominant left eigenvalue of the Transfer Matrix is not yet 1 within tolerance.')
            print('We are calculating it again with the new estimate for the dominant left eigenvector.')
            norm, Gl = mps._canonical_form_dominant_gram_matrix(i1, True, tol_xi, Gl)
        if abs(1. - norm) > 1.e-13:
            warnings.warn("Although we renormalized the TransferMatrix, "
                          "the largest eigenvalue is not 1")  # (this shouldn't happen)
        mps._B[i1] /= np.sqrt(norm)  # correct norm again
        if not renormalize:
            mps.norm *= np.sqrt(norm)
        # bring bond to canonical form
        mps, Gl, Wr = my_canonical_form_correct_left(mps, i1, Gl, Wr, chi_max = chi_list[i1 % L])
        # now the bond (i1-1,i1) is in canonical form

        # phase 2: sweep from right to left; find other right eigenvectors and make them diagonal
        Wr_list[i1] = Wr  # diag(Wr) is right eigenvector on bond (i1-1, i1)
        for j1 in range(i1 - 1, i1 - L, -1):
            B1 = mps.get_B(j1, 'B')
            axes = [mps._p_label + ['vR'], mps._get_p_label('*') + ['vR*']]
            Gr = npc.tensordot(B1.scale_axis(Wr, 'vR'), B1.conj(), axes=axes)
            Wr = mps._canonical_form_correct_right(j1, Gr)
            Wr_list[j1 % L] = Wr

        # phase 3: sweep from left to right; find other left eigenvectors,
        # bring each bond into canonical form
        for j1 in range(i1 - L + 1, i1, +1):
            # find Gl on bond j1-1, j1
            B1 = mps.get_B(j1 - 1, 'B')
            Gl = npc.tensordot(
                B1.conj(),  # old B1; now on site j1-1
                npc.tensordot(Gl, B1, axes=['vR', 'vL']),
                axes=[mps._get_p_label('*') + ['vL*'], mps._p_label + ['vR*']])
            # axes=[['p*', 'vL*'], ['p', 'vR*']])
            mps, Gl, Wr = my_canonical_form_correct_left(mps, j1, Gl, Wr_list[j1 % L], chi_max = chi_list[j1 % L])
        
        return mps


def my_canonical_form_correct_left(mps, i1, Gl, Wr, chi_max, eps=2. * np.finfo(np.double).eps):
        """Bring into canonical form on bond (i0, i1) where i0= i1 - 1.

        Given the left Gram matrix Gl (with legs 'vR*', 'vR')
        and right diag(Wr), compute and diagonalize the density matrix
        ``rho = sqrt(Wr) Gl sqrt(Wr) -> Y^H S^2 Y`` (Y acting on ket, Y^H on bra).
        Then we can update
        B[i0] -> B[i0] sqrt(Wr) Y^H
        B[i1] -> Y 1/sqrt(Wr) B[i1]
        Thus the new dominant left eigenvector is
        Gl -> Y sqrt(Wr) Gl sqrt(Wr) Y^H = S^2
        and dominant right eigenvector is
        diag(Wr) -> Y 1/sqrt(Wr) diag(Wr) 1/sqrt(W) Y^H = diag(1)
        i.e., we brought the bond to canonical form and `S` is the Schmidt spectrum.
        """
        sqrt_Wr = np.sqrt(Wr)
        Gl.itranspose(['vR*', 'vR'])
        rhor = Gl.scale_axis(sqrt_Wr, 0).iscale_axis(sqrt_Wr, 1)
        S2, YH = npc.eigh(rhor, sort='>')  # YH has legs 'vR*', 'vR'
        S2 /= np.sum(S2)  # equivalent to normalizing tr(rhor)=1
        s_norm = 1.
        L = mps.L
        # Truncate Schmidt values s.t. chi <= chi_max
        truncate = (np.arange(len(S2)) < chi_max)
        if np.count_nonzero(truncate) < len(S2):
            YH.iproject(truncate, axes=1)
            S2 = S2[truncate]
            print('Truncation of bond ({}, {}) in myCanonical_form_infinite to chi={}.'.format((i1-1) % L, i1 % L, len(S2)))
            s_norm = np.sqrt(np.sum(S2))
        # discard small values on order of machine precision
        proj = (S2 > eps)
        if np.count_nonzero(proj) < len(S2):
            # project into non-degenerate subspace, reducing the bond dimensions!
            warnings.warn("canonical_form_infinite: project to smaller bond dimension",
                          stacklevel=2)
            YH.iproject(proj, axes=1)
            S2 = S2[proj]
            s_norm = np.sqrt(np.sum(S2))
        S = np.sqrt(S2) / s_norm
        mps.set_SL(i1, S)
        Yl = YH.scale_axis(sqrt_Wr / s_norm, 0).iset_leg_labels(['vL', 'vR'])
        Yr = YH.transpose().iconj().scale_axis(1. / sqrt_Wr, 1).iset_leg_labels(['vL', 'vR'])
        i0 = i1 - 1
        mps.set_B(i0, npc.tensordot(mps.get_B(i0), Yl, axes=['vR', 'vL']))
        mps.set_B(i1, npc.tensordot(Yr, mps.get_B(i1), axes=['vR', 'vL']))
        Gl = npc.tensordot(Gl, Yl, axes=['vR', 'vL'])
        Gl = npc.tensordot(Yl.conj(), Gl, axes=['vL*', 'vR*'])  # labels 'vR*', 'vR'
        Gl /= npc.trace(Gl)
        # Gl is diag(S**2) up to numerical errors...
        return mps, Gl, np.ones(Yr.legs[0].ind_len, np.float)
