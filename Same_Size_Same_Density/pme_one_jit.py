from typing import Tuple, Optional
from typing import Iterable, Tuple, Optional
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, jit
from jax.scipy.special import erf, erfc

# from dmff.settings import DO_JIT
# from dmff.common.constants import DIELECTRIC
# from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
# from dmff.admp.settings import POL_CONV, MAX_N_POL
# from dmff.admp.recip import generate_pme_recip, Ck_1
from dmff.admp.multipole import (
    C1_c2h,
    convert_cart2harm,
    rot_ind_global2local,
    rot_global2local,
    rot_local2global
)
from dmff.admp.spatial import (
    v_pbc_shift,
    generate_construct_local_frames,
    build_quasi_internal
)
from dmff.admp.pairwise import (
    distribute_scalar,
    distribute_v3,
    distribute_multipoles,
    distribute_matrix
)
import timeit
import logging
DO_JIT = True
DIELECTRIC =1389.3545584


DEFAULT_THOLE_WIDTH = 5.0
def jit_condition(*args, **kwargs):
    def jit_deco(func):
        if DO_JIT:
            return jit(func, *args, **kwargs)
        else:
            return func

    return jit_deco
@vmap
def regularize_pairs(p):
    dp = p[1] - p[0]
    dp = jnp.piecewise(dp, (dp <= 0, dp > 0),
                       (lambda x: jnp.array(1), lambda x: jnp.array(0)))
    dp_vec = jnp.array([dp, 2 * dp])
    p = p - dp_vec
    return p
# @jit
@vmap
def regularize_pairs_new(p):
    dp = p[1] - p[0]
    dp = jax.lax.cond(dp <= 0, lambda _: jnp.array(1), lambda _: jnp.array(0), None)
    dp_vec = jnp.array([dp, 2 * dp])
    p = p - dp_vec
    return p
@vmap
def pair_buffer_scales(p):
    return jnp.piecewise(p[0] - p[1], (p[0] - p[1] < 0, p[0] - p[1] >= 0),
                         (lambda x: jnp.array(1), lambda x: jnp.array(0)))
@jit
@vmap
def pair_buffer_scales_new(p):
    dp = p[0] - p[1]
    result = jax.lax.cond(jax.lax.lt(dp, 0), lambda _: jnp.array(1), lambda _: jnp.array(0), None)
    return result

def Ck_1(ksq, kappa, V):
    return 2*jnp.pi/V/ksq * jnp.exp(-ksq/4/kappa**2)
def generate_pme_recip_new(Ck_fn, kappa, gamma, pme_order, K1, K2, K3, lmax):
    # Currently only supports pme_order=6
    # Because only the 6-th order spline function is hard implemented
    pme_order = 6
    # global variables for the reciprocal module, all related to pme_order
    bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    n_mesh = pme_order ** 3
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))

    def pme_recip(positions, box, Q):
        '''
        The generated pme_recip space calculator
        kappa, pme_order, K1, K2, K3, and lmax are passed and fixed when the calculator is generated
        '''

        def get_recip_vectors(N, box):
            """
            Computes reciprocal lattice vectors of the grid

            Input:
                N:
                    (3,)-shaped array, (K1, K2, K3)
                box:
                    3 x 3 matrix, box parallelepiped vectors arranged in TODO rows or columns?

            Output:
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)
            """
            Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
            return Nj_Aji_star

        def u_reference(R_a, Nj_Aji_star):
            """
            Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid.
            This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates,
            and the corresponding values of xyz fractional displacements from real coordinate to the reference point.

            Inputs:
                R_a:
                    N_a * 3 matrix containing positions of sites
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)

            Outputs:
                m_u0:
                    N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
                u0:
                    N_a * 3 matrix, (R_a - R_m)*a_star values
            """
            R_in_m_basis = jnp.einsum("ij,kj->ki", Nj_Aji_star, R_a)
            m_u0 = jnp.ceil(R_in_m_basis).astype(int)
            u0 = (m_u0 - R_in_m_basis) + pme_order / 2
            return m_u0, u0

        def bspline_new(u, order=pme_order):
            """
            Computes the cardinal B-spline function
            """
            if order == 6:
                u2 = u ** 2
                u3 = u ** 3
                u4 = u ** 4
                u5 = u ** 5
                u_less_1 = u - 1
                u_less_1_p5 = u_less_1 ** 5
                u_less_2 = u - 2
                u_less_2_p5 = u_less_2 ** 5
                u_less_3 = u - 3
                u_less_3_p5 = u_less_3 ** 5
                conditions = [
                    jnp.logical_and(u >= 0., u < 1.),
                    jnp.logical_and(u >= 1., u < 2.),
                    jnp.logical_and(u >= 2., u < 3.),
                    jnp.logical_and(u >= 3., u < 4.),
                    jnp.logical_and(u >= 4., u < 5.),
                    jnp.logical_and(u >= 5., u < 6.)
                ]
                outputs = [
                    u5 / 120,
                    u5 / 120 - u_less_1_p5 / 20,
                    u5 / 120 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
                    u5 / 120 - u_less_3_p5 / 6 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
                    u5 / 24 - u4 + 19 * u3 / 2 - 89 * u2 / 2 + 409 * u / 4 - 1829 / 20,
                    -u5 / 120 + u4 / 4 - 3 * u3 + 18 * u2 - 54 * u + 324 / 5
                ]
                return jnp.sum(jnp.stack([condition * output for condition, output in zip(conditions, outputs)]),
                               axis=0)

        def bspline(u, order=pme_order):
            """
            Computes the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 5 / 120,
                        lambda u: u ** 5 / 120 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 120 + (u - 2) ** 5 / 8 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 120 - (u - 3) ** 5 / 6 + (u - 2) ** 5 / 8 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 24 - u ** 4 + 19 * u ** 3 / 2 - 89 * u ** 2 / 2 + 409 * u / 4 - 1829 / 20,
                        lambda u: -u ** 5 / 120 + u ** 4 / 4 - 3 * u ** 3 + 18 * u ** 2 - 54 * u + 324 / 5
                    ]
                )

        def bspline_prime(u, order=pme_order):
            """
            Computes first derivative of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 4 / 24,
                        lambda u: u ** 4 / 24 - (u - 1) ** 4 / 4,
                        lambda u: u ** 4 / 24 + 5 * (u - 2) ** 4 / 8 - (u - 1) ** 4 / 4,
                        lambda u: -5 * u ** 4 / 12 + 6 * u ** 3 - 63 * u ** 2 / 2 + 71 * u - 231 / 4,
                        lambda u: 5 * u ** 4 / 24 - 4 * u ** 3 + 57 * u ** 2 / 2 - 89 * u + 409 / 4,
                        lambda u: -u ** 4 / 24 + u ** 3 - 9 * u ** 2 + 36 * u - 54
                    ]
                )

        def bspline_prime2(u, order=pme_order):
            """
            Computes second derivate of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 3 / 6,
                        lambda u: u ** 3 / 6 - (u - 1) ** 3,
                        lambda u: 5 * u ** 3 / 3 - 12 * u ** 2 + 27 * u - 19,
                        lambda u: -5 * u ** 3 / 3 + 18 * u ** 2 - 63 * u + 71,
                        lambda u: 5 * u ** 3 / 6 - 12 * u ** 2 + 57 * u - 89,
                        lambda u: -u ** 3 / 6 + 3 * u ** 2 - 18 * u + 36
                    ]
                )

        def theta_eval(u, M_u):
            """
            Evaluates the value of theta given 3D u values at ... points

            Input:
                u:
                    ... x 3 matrix

            Output:
                theta:
                    ... matrix
            """
            theta = jnp.prod(M_u, axis=-1)
            return theta

        def thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u):
            """
            First derivative of theta with respect to x,y,z directions

            Input:
                u
                Nj_Aji_star:
                    reciprocal lattice vectors

            Output:
                N_a * 3 matrix
            """

            div = jnp.array([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T

            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("ij,kj->ki", -Nj_Aji_star, div)

        def theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
            """
            compute the 3 x 3 second derivatives of theta with respect to xyz

            Input:
                u
                Nj_Aji_star

            Output:
                N_A * 3 * 3
            """

            div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
            div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
            div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]

            div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
            div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
            div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]

            div_10 = div_01
            div_20 = div_02
            div_21 = div_12

            div = jnp.array([
                [div_00, div_01, div_02],
                [div_10, div_11, div_12],
                [div_20, div_21, div_22],
            ]).swapaxes(0, 2)

            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

        def sph_harmonics_GO(u0, Nj_Aji_star):
            '''
            Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
            00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
            Currently supports lmax <= 2

            Inputs:
                u0:
                    a N_a * 3 matrix containing all positions
                Nj_Aji_star:
                    reciprocal lattice vectors in the m-grid
                lmax:
                    int: max L

            Output:
                harmonics:
                    a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                    evaluated at 6*6*6 integer points about reference points m_u0
            '''

            n_harm = int((lmax + 1) ** 2)

            N_a = u0.shape[0]
            # mesh points around each site
            u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a * n_mesh, 3))

            # M_u = bspline(u)
            M_u = bspline_new(u)
            theta = theta_eval(u, M_u)
            if lmax == 0:
                return theta.reshape(N_a, n_mesh, n_harm)

            # dipole
            Mprime_u = bspline_prime(u)
            thetaprime = thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u)
            harmonics_1 = jnp.stack(
                [theta,
                 thetaprime[:, 2],
                 thetaprime[:, 0],
                 thetaprime[:, 1]],
                axis=-1
            )

            if lmax == 1:
                return harmonics_1.reshape(N_a, n_mesh, n_harm)

            # quadrapole
            M2prime_u = bspline_prime2(u)
            theta2prime = theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
            rt3 = jnp.sqrt(3)
            harmonics_2 = jnp.hstack(
                [harmonics_1,
                 jnp.stack([(3 * theta2prime[:, 2, 2] - jnp.trace(theta2prime, axis1=1, axis2=2)) / 2,
                            rt3 * theta2prime[:, 0, 2],
                            rt3 * theta2prime[:, 1, 2],
                            rt3 / 2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
                            rt3 * theta2prime[:, 0, 1]], axis=1)]
            )
            if lmax == 2:
                return harmonics_2.reshape(N_a, n_mesh, n_harm)
            else:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

        def Q_m_peratom(Q, sph_harms):
            """
            Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983

            Inputs:
                Q:
                    N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
                sph_harms:
                    N_a, 216, (l+1)**2
                lmax:
                    int: maximal L

            Output:
                Q_m_pera:
                    N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
            """

            N_a = sph_harms.shape[0]

            if lmax > 2:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

            Q_dbf = Q[:, 0:1]

            if lmax >= 1:
                Q_dbf = jnp.hstack([Q_dbf, Q[:, 1:4]])
            if lmax >= 2:
                Q_dbf = jnp.hstack([Q_dbf, Q[:, 4:9] / 3])

            Q_m_pera = jnp.sum(Q_dbf[:, jnp.newaxis, :] * sph_harms, axis=2)

            assert Q_m_pera.shape == (N_a, n_mesh)
            return Q_m_pera

        def Q_mesh_on_m(Q_mesh_pera, m_u0, N):
            """
            Reduce the local Q_m_peratom into the global mesh

            Input:
                Q_mesh_pera, m_u0, N

            Output:
                Q_mesh:
                    Nx * Ny * Nz matrix
            """
            indices_arr = jnp.mod(m_u0[:, np.newaxis, :] + shifts, N[np.newaxis, np.newaxis, :])
            ### jax trick implementation without using for loop
            ### NOTICE: this implementation does not work with numpy!
            Q_mesh = jnp.zeros((N[0], N[1], N[2]))
            Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)
            return Q_mesh

        def setup_kpts_integer(N):
            """
            Outputs:
                kpts_int:
                    n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
            """
            N_half = N.reshape(3)
            kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2), - (N_half[i] - 1) // 2) for
                          i in range(3)]
            kpts_int = jnp.hstack([ki.flatten()[:, jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
            return kpts_int

        def setup_kpts(box, kpts_int):
            '''
            This function sets up the k-points used for reciprocal space calculations

            Input:
                box:
                    3 * 3, three axis arranged in rows
                kpts_int:
                    n_k * 3 matrix

            Output:
                kpts:
                    4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
            '''
            # in this array, a*, b*, c* (without 2*pi) are arranged in column
            box_inv = jnp.linalg.inv(box)
            # K * 3, coordinate in reciprocal space
            kpts = 2 * jnp.pi * kpts_int.dot(box_inv)
            ksq = jnp.sum(kpts ** 2, axis=1)
            # 4 * K
            kpts = jnp.hstack((kpts, ksq[:, jnp.newaxis])).T
            # kpts = jnp.column_stack((kpts, ksq))
            # kpts = jnp.column_stack((kpts, ksq)).T
            return kpts

        def spread_Q(positions, box, Q):
            '''
            This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

            Input:
                positions:
                    Na * 3: positions of each site
                box:
                    3 * 3: box
                Q:
                    Na * (lmax+1)**2: the multipole of each site in global frame

            Output:
                Q_mesh:
                    K1 * K2 * K3: the meshed multipoles

            '''
            Nj_Aji_star = get_recip_vectors(N, box)
            # For each atom, find the reference mesh point, and u position of the site
            m_u0, u0 = u_reference(positions, Nj_Aji_star)
            # find out the STGO values of each grid point
            sph_harms = sph_harmonics_GO(u0, Nj_Aji_star)
            # find out the local meshed values for each site
            Q_mesh_pera = Q_m_peratom(Q, sph_harms)
            return Q_mesh_on_m(Q_mesh_pera, m_u0, N)

        # spread Q
        N = np.array([K1, K2, K3])
        Q_mesh = spread_Q(positions, box, Q)
        N = N.reshape(1, 1, 3)
        kpts_int = setup_kpts_integer(N)
        kpts = setup_kpts(box, kpts_int)
        m = jnp.linspace(-pme_order // 2 + 1, pme_order // 2 - 1, pme_order - 1).reshape(pme_order - 1, 1, 1)
        # m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
        theta_k = jnp.prod(
            jnp.sum(
                bspline_new(m + pme_order / 2) * jnp.cos(2 * jnp.pi * m * kpts_int[jnp.newaxis] / N),
                axis=0
            ),
            axis=1
        )
        V = jnp.linalg.det(box)
        S_k = jnp.fft.fftn(Q_mesh).flatten()
        # for electrostatic, need to exclude gamma point
        # for dispersion, need to include gamma point
        if not gamma:
            C_k = Ck_fn(kpts[3, 1:], kappa, V)
            E_k = C_k * jnp.abs(S_k[1:] / theta_k[1:]) ** 2
        else:
            C_k = Ck_fn(kpts[3, :], kappa, V)
            # debug
            # for i in range(1000):
            #     print('%15.8f%15.8f'%(jnp.real(C_k[i]), jnp.imag(C_k[i])))
            E_k = C_k * jnp.abs(S_k / theta_k) ** 2

        if not gamma:  # doing electrics
            return jnp.sum(E_k) * DIELECTRIC
        else:
            return jnp.sum(E_k)

    if DO_JIT:
        return jit(pme_recip, static_argnums=())
    else:
        return pme_recip

def generate_pme_recip_moveN(Ck_fn, kappa, gamma, pme_order, K1, K2, K3, lmax):
    # Currently only supports pme_order=6
    # Because only the 6-th order spline function is hard implemented
    pme_order = 6
    # global variables for the reciprocal module, all related to pme_order
    bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    n_mesh = pme_order ** 3
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
    N = np.array([K1, K2, K3]).reshape(1,1,3)
    def pme_recip(positions, box, Q):
        '''
        The generated pme_recip space calculator
        kappa, pme_order, K1, K2, K3, and lmax are passed and fixed when the calculator is generated
        '''

        def get_recip_vectors(N, box):
            """
            Computes reciprocal lattice vectors of the grid

            Input:
                N:
                    (3,)-shaped array, (K1, K2, K3)
                box:
                    3 x 3 matrix, box parallelepiped vectors arranged in TODO rows or columns?

            Output:
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)
            """
            Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
            return Nj_Aji_star

        def u_reference(R_a, Nj_Aji_star):
            """
            Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid.
            This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates,
            and the corresponding values of xyz fractional displacements from real coordinate to the reference point.

            Inputs:
                R_a:
                    N_a * 3 matrix containing positions of sites
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)

            Outputs:
                m_u0:
                    N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
                u0:
                    N_a * 3 matrix, (R_a - R_m)*a_star values
            """
            R_in_m_basis = jnp.einsum("ij,kj->ki", Nj_Aji_star, R_a)
            m_u0 = jnp.ceil(R_in_m_basis).astype(int)
            u0 = (m_u0 - R_in_m_basis) + pme_order / 2
            return m_u0, u0

        def bspline_new(u, order=pme_order):
            """
            Computes the cardinal B-spline function
            """
            if order == 6:
                u2 = u ** 2
                u3 = u ** 3
                u4 = u ** 4
                u5 = u ** 5
                u_less_1 = u - 1
                u_less_1_p5 = u_less_1 ** 5
                u_less_2 = u - 2
                u_less_2_p5 = u_less_2 ** 5
                u_less_3 = u - 3
                u_less_3_p5 = u_less_3 ** 5
                conditions = [
                    jnp.logical_and(u >= 0., u < 1.),
                    jnp.logical_and(u >= 1., u < 2.),
                    jnp.logical_and(u >= 2., u < 3.),
                    jnp.logical_and(u >= 3., u < 4.),
                    jnp.logical_and(u >= 4., u < 5.),
                    jnp.logical_and(u >= 5., u < 6.)
                ]
                outputs = [
                    u5 / 120,
                    u5 / 120 - u_less_1_p5 / 20,
                    u5 / 120 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
                    u5 / 120 - u_less_3_p5 / 6 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
                    u5 / 24 - u4 + 19 * u3 / 2 - 89 * u2 / 2 + 409 * u / 4 - 1829 / 20,
                    -u5 / 120 + u4 / 4 - 3 * u3 + 18 * u2 - 54 * u + 324 / 5
                ]
                return jnp.sum(jnp.stack([condition * output for condition, output in zip(conditions, outputs)]),
                               axis=0)

        def bspline(u, order=pme_order):
            """
            Computes the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 5 / 120,
                        lambda u: u ** 5 / 120 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 120 + (u - 2) ** 5 / 8 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 120 - (u - 3) ** 5 / 6 + (u - 2) ** 5 / 8 - (u - 1) ** 5 / 20,
                        lambda u: u ** 5 / 24 - u ** 4 + 19 * u ** 3 / 2 - 89 * u ** 2 / 2 + 409 * u / 4 - 1829 / 20,
                        lambda u: -u ** 5 / 120 + u ** 4 / 4 - 3 * u ** 3 + 18 * u ** 2 - 54 * u + 324 / 5
                    ]
                )

        def bspline_prime(u, order=pme_order):
            """
            Computes first derivative of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 4 / 24,
                        lambda u: u ** 4 / 24 - (u - 1) ** 4 / 4,
                        lambda u: u ** 4 / 24 + 5 * (u - 2) ** 4 / 8 - (u - 1) ** 4 / 4,
                        lambda u: -5 * u ** 4 / 12 + 6 * u ** 3 - 63 * u ** 2 / 2 + 71 * u - 231 / 4,
                        lambda u: 5 * u ** 4 / 24 - 4 * u ** 3 + 57 * u ** 2 / 2 - 89 * u + 409 / 4,
                        lambda u: -u ** 4 / 24 + u ** 3 - 9 * u ** 2 + 36 * u - 54
                    ]
                )

        def bspline_prime2(u, order=pme_order):
            """
            Computes second derivate of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(
                    u,
                    [
                        jnp.logical_and(u >= 0., u < 1.),
                        jnp.logical_and(u >= 1., u < 2.),
                        jnp.logical_and(u >= 2., u < 3.),
                        jnp.logical_and(u >= 3., u < 4.),
                        jnp.logical_and(u >= 4., u < 5.),
                        jnp.logical_and(u >= 5., u < 6.)
                    ],
                    [
                        lambda u: u ** 3 / 6,
                        lambda u: u ** 3 / 6 - (u - 1) ** 3,
                        lambda u: 5 * u ** 3 / 3 - 12 * u ** 2 + 27 * u - 19,
                        lambda u: -5 * u ** 3 / 3 + 18 * u ** 2 - 63 * u + 71,
                        lambda u: 5 * u ** 3 / 6 - 12 * u ** 2 + 57 * u - 89,
                        lambda u: -u ** 3 / 6 + 3 * u ** 2 - 18 * u + 36
                    ]
                )

        def theta_eval(u, M_u):
            """
            Evaluates the value of theta given 3D u values at ... points

            Input:
                u:
                    ... x 3 matrix

            Output:
                theta:
                    ... matrix
            """
            theta = jnp.prod(M_u, axis=-1)
            return theta

        def thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u):
            """
            First derivative of theta with respect to x,y,z directions

            Input:
                u
                Nj_Aji_star:
                    reciprocal lattice vectors

            Output:
                N_a * 3 matrix
            """

            div = jnp.array([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T

            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("ij,kj->ki", -Nj_Aji_star, div)

        def theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
            """
            compute the 3 x 3 second derivatives of theta with respect to xyz

            Input:
                u
                Nj_Aji_star

            Output:
                N_A * 3 * 3
            """

            div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
            div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
            div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]

            div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
            div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
            div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]

            div_10 = div_01
            div_20 = div_02
            div_21 = div_12

            div = jnp.array([
                [div_00, div_01, div_02],
                [div_10, div_11, div_12],
                [div_20, div_21, div_22],
            ]).swapaxes(0, 2)

            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

        def sph_harmonics_GO(u0, Nj_Aji_star):
            '''
            Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
            00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
            Currently supports lmax <= 2

            Inputs:
                u0:
                    a N_a * 3 matrix containing all positions
                Nj_Aji_star:
                    reciprocal lattice vectors in the m-grid
                lmax:
                    int: max L

            Output:
                harmonics:
                    a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                    evaluated at 6*6*6 integer points about reference points m_u0
            '''

            n_harm = int((lmax + 1) ** 2)

            N_a = u0.shape[0]
            # mesh points around each site
            u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a * n_mesh, 3))

            # M_u = bspline(u)
            M_u = bspline_new(u)
            theta = theta_eval(u, M_u)
            if lmax == 0:
                return theta.reshape(N_a, n_mesh, n_harm)

            # dipole
            Mprime_u = bspline_prime(u)
            thetaprime = thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u)
            harmonics_1 = jnp.stack(
                [theta,
                 thetaprime[:, 2],
                 thetaprime[:, 0],
                 thetaprime[:, 1]],
                axis=-1
            )

            if lmax == 1:
                return harmonics_1.reshape(N_a, n_mesh, n_harm)

            # quadrapole
            M2prime_u = bspline_prime2(u)
            theta2prime = theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
            rt3 = jnp.sqrt(3)
            harmonics_2 = jnp.hstack(
                [harmonics_1,
                 jnp.stack([(3 * theta2prime[:, 2, 2] - jnp.trace(theta2prime, axis1=1, axis2=2)) / 2,
                            rt3 * theta2prime[:, 0, 2],
                            rt3 * theta2prime[:, 1, 2],
                            rt3 / 2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
                            rt3 * theta2prime[:, 0, 1]], axis=1)]
            )
            if lmax == 2:
                return harmonics_2.reshape(N_a, n_mesh, n_harm)
            else:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

        def Q_m_peratom(Q, sph_harms):
            """
            Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983

            Inputs:
                Q:
                    N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
                sph_harms:
                    N_a, 216, (l+1)**2
                lmax:
                    int: maximal L

            Output:
                Q_m_pera:
                    N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
            """

            N_a = sph_harms.shape[0]

            if lmax > 2:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

            Q_dbf = Q[:, 0:1]

            if lmax >= 1:
                Q_dbf = jnp.hstack([Q_dbf, Q[:, 1:4]])
            if lmax >= 2:
                Q_dbf = jnp.hstack([Q_dbf, Q[:, 4:9] / 3])

            Q_m_pera = jnp.sum(Q_dbf[:, jnp.newaxis, :] * sph_harms, axis=2)

            assert Q_m_pera.shape == (N_a, n_mesh)
            return Q_m_pera

        def Q_mesh_on_m(Q_mesh_pera, m_u0, N):
            """
            Reduce the local Q_m_peratom into the global mesh

            Input:
                Q_mesh_pera, m_u0, N

            Output:
                Q_mesh:
                    Nx * Ny * Nz matrix
            """
            indices_arr = jnp.mod(m_u0[:, np.newaxis, :] + shifts, N[np.newaxis, np.newaxis, :])
            ### jax trick implementation without using for loop
            ### NOTICE: this implementation does not work with numpy!
            Q_mesh = jnp.zeros((N[0], N[1], N[2]))
            Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)
            return Q_mesh

        def setup_kpts_integer(N):
            """
            Outputs:
                kpts_int:
                    n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
            """
            N_half = N.reshape(3)
            kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2), - (N_half[i] - 1) // 2) for
                          i in range(3)]
            kpts_int = jnp.hstack([ki.flatten()[:, jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
            return kpts_int

        def setup_kpts(box, kpts_int):
            '''
            This function sets up the k-points used for reciprocal space calculations

            Input:
                box:
                    3 * 3, three axis arranged in rows
                kpts_int:
                    n_k * 3 matrix

            Output:
                kpts:
                    4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
            '''
            # in this array, a*, b*, c* (without 2*pi) are arranged in column
            box_inv = jnp.linalg.inv(box)
            # K * 3, coordinate in reciprocal space
            kpts = 2 * jnp.pi * kpts_int.dot(box_inv)
            ksq = jnp.sum(kpts ** 2, axis=1)
            # 4 * K
            kpts = jnp.hstack((kpts, ksq[:, jnp.newaxis])).T
            # kpts = jnp.column_stack((kpts, ksq))
            # kpts = jnp.column_stack((kpts, ksq)).T
            return kpts

        def spread_Q(positions, box, Q):
            '''
            This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

            Input:
                positions:
                    Na * 3: positions of each site
                box:
                    3 * 3: box
                Q:
                    Na * (lmax+1)**2: the multipole of each site in global frame

            Output:
                Q_mesh:
                    K1 * K2 * K3: the meshed multipoles

            '''
            Nj_Aji_star = get_recip_vectors(N, box)
            # For each atom, find the reference mesh point, and u position of the site
            m_u0, u0 = u_reference(positions, Nj_Aji_star)
            # find out the STGO values of each grid point
            sph_harms = sph_harmonics_GO(u0, Nj_Aji_star)
            # find out the local meshed values for each site
            Q_mesh_pera = Q_m_peratom(Q, sph_harms)
            return Q_mesh_on_m(Q_mesh_pera, m_u0, N)

        # spread Q
        N = np.array([K1, K2, K3])
        Q_mesh = spread_Q(positions, box, Q)
        N = N.reshape(1, 1, 3)
        kpts_int = setup_kpts_integer(N)
        kpts = setup_kpts(box, kpts_int)
        m = jnp.linspace(-pme_order // 2 + 1, pme_order // 2 - 1, pme_order - 1).reshape(pme_order - 1, 1, 1)
        # m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
        theta_k = jnp.prod(
            jnp.sum(
                bspline_new(m + pme_order / 2) * jnp.cos(2 * jnp.pi * m * kpts_int[jnp.newaxis] / N),
                axis=0
            ),
            axis=1
        )
        V = jnp.linalg.det(box)
        S_k = jnp.fft.fftn(Q_mesh).flatten()
        # for electrostatic, need to exclude gamma point
        # for dispersion, need to include gamma point
        if not gamma:
            C_k = Ck_fn(kpts[3, 1:], kappa, V)
            E_k = C_k * jnp.abs(S_k[1:] / theta_k[1:]) ** 2
        else:
            C_k = Ck_fn(kpts[3, :], kappa, V)
            # debug
            # for i in range(1000):
            #     print('%15.8f%15.8f'%(jnp.real(C_k[i]), jnp.imag(C_k[i])))
            E_k = C_k * jnp.abs(S_k / theta_k) ** 2

        if not gamma:  # doing electrics
            return jnp.sum(E_k) * DIELECTRIC
        else:
            return jnp.sum(E_k)

    if DO_JIT:
        return jit(pme_recip, static_argnums=())
    else:
        return pme_recip
class CoulombPMEForce:

    def __init__(
            self,
            r_cut: float,
            map_prm: Iterable[int],
            kappa: float,
            K: Tuple[int, int, int],
            pme_order: int = 6,
            topology_matrix: Optional[jnp.array] = None,
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.lmax = 0
        self.kappa = kappa
        self.K1, self.K2, self.K3 = K[0], K[1], K[2]
        self.pme_order = pme_order
        self.top_mat = topology_matrix
        assert pme_order == 6, "PME order other than 6 is not supported"

    def generate_get_energy(self):

        def get_energy(positions, box, pairs, charges, mscales):

            pme_recip_fn = generate_pme_recip_new(
                Ck_fn=Ck_1,
                kappa=self.kappa / 10,
                gamma=False,
                pme_order=self.pme_order,
                K1=self.K1,
                K2=self.K2,
                K3=self.K3,
                lmax=self.lmax,
            )

            atomCharges = charges[self.map_prm[np.arange(positions.shape[0])]]
            atomChargesT = jnp.reshape(atomCharges, (-1, 1))

            return energy_pme(
                positions * 10,
                box * 10,
                pairs,
                atomChargesT,
                None,
                None,
                None,
                mscales,
                None,
                None,
                None,
                pme_recip_fn,
                self.kappa / 10,
                self.K1,
                self.K2,
                self.K3,
                self.lmax,
                False,
            )

        def get_energy_bcc(positions, box, pairs, pre_charges, bcc, mscales):
            charges = pre_charges + jnp.dot(self.top_mat, bcc).flatten()
            return get_energy(positions, box, pairs, charges, mscales)

        if self.top_mat is None:
            return get_energy
        else:
            return get_energy_bcc

class CoulombPMEForce_MoveN:

    def __init__(
            self,
            r_cut: float,
            map_prm: Iterable[int],
            kappa: float,
            K: Tuple[int, int, int],
            pme_order: int = 6,
            topology_matrix: Optional[jnp.array] = None,
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.lmax = 0
        self.kappa = kappa
        self.K1, self.K2, self.K3 = K[0], K[1], K[2]
        self.pme_order = pme_order
        self.top_mat = topology_matrix
        assert pme_order == 6, "PME order other than 6 is not supported"

    def generate_get_energy(self):

        def get_energy(positions, box, pairs, charges, mscales):

            pme_recip_fn = generate_pme_recip_moveN(
                Ck_fn=Ck_1,
                kappa=self.kappa / 10,
                gamma=False,
                pme_order=self.pme_order,
                K1=self.K1,
                K2=self.K2,
                K3=self.K3,
                lmax=self.lmax,
            )

            atomCharges = charges[self.map_prm[np.arange(positions.shape[0])]]
            atomChargesT = jnp.reshape(atomCharges, (-1, 1))

            return energy_pme(
                positions * 10,
                box * 10,
                pairs,
                atomChargesT,
                None,
                None,
                None,
                mscales,
                None,
                None,
                None,
                pme_recip_fn,
                self.kappa / 10,
                self.K1,
                self.K2,
                self.K3,
                self.lmax,
                False,
            )

        def get_energy_bcc(positions, box, pairs, pre_charges, bcc, mscales):
            charges = pre_charges + jnp.dot(self.top_mat, bcc).flatten()
            return get_energy(positions, box, pairs, charges, mscales)

        if self.top_mat is None:
            return get_energy
        else:
            return get_energy_bcc


@jit_condition(static_argnums=(7))
def calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax=2):
    r'''
    This function calculates the eUindCoefs at once
       ## compute the Thole damping factors for energies
     eUindCoefs is basically the interaction tensor between permanent multipole components and induced dipoles
    Everything should be done in the so called quasi-internal (qi) frame


    Inputs:
        dr:
            float: distance between one pair of particles
        dmp
            float: damping factors between one pair of particles
        mscales:
            float: scaling factor between permanent - permanent multipole interactions, for each pair
        pscales:
            float: scaling factor between permanent - induced multipole interactions, for each pair
        au:
            float: for damping factors
        kappa:
            float: \kappa in PME, unit in A^-1
        lmax:
            int: max L

    Output:
        Interaction tensors components
    '''
    ## pscale == 0 ? thole1 + thole2 : DEFAULT_THOLE_WIDTH
    w = jnp.heaviside(pscales, 0)
    a = w * DEFAULT_THOLE_WIDTH + (1 - w) * (thole1 + thole2)

    dmp = trim_val_0(dmp)
    u = trim_val_infty(dr / dmp)

    ## au <= 50 aupi = au ;au> 50 aupi = 50
    au = a * u
    expau = jnp.piecewise(au, [au < 50, au >= 50], [lambda au: jnp.exp(-au), lambda au: jnp.array(0)])

    ## compute the Thole damping factors for energies
    au2 = trim_val_infty(au * au)
    au3 = trim_val_infty(au2 * au)
    au4 = trim_val_infty(au3 * au)
    au5 = trim_val_infty(au4 * au)
    au6 = trim_val_infty(au5 * au)

    ##  Thole damping factors for energies
    thole_c = 1.0 - expau * (1.0 + au + 0.5 * au2)
    thole_d0 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 4.0)
    thole_d1 = 1.0 - expau * (1.0 + au + 0.5 * au2)
    thole_q0 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 6.0 + au4 / 18.0)
    thole_q1 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 6.0)
    # copied from calc_e_perm
    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC * (rInv ** i) for i in range(0, 9)])
    alphaRVec = jnp.array([(kappa * dr) ** i for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])

    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i - 1] + (tmp * X / doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

    ## C-Uind
    cud = 2.0 * rInvVec[2] * (pscales * thole_c + bVec[2])
    if lmax >= 1:
        ##  D-Uind terms
        dud_m0 = -2.0 * 2.0 / 3.0 * rInvVec[3] * (3.0 * (pscales * thole_d0 + bVec[3]) + alphaRVec[3] * X)
        dud_m1 = 2.0 * rInvVec[3] * (pscales * thole_d1 + bVec[3] - 2.0 / 3.0 * alphaRVec[3] * X)
    else:
        dud_m0 = 0.0
        dud_m1 = 0.0

    if lmax >= 2:
        ## Uind-Q
        udq_m0 = 2.0 * rInvVec[4] * (3.0 * (pscales * thole_q0 + bVec[3]) + 4 / 3 * alphaRVec[5] * X)
        udq_m1 = -2.0 * jnp.sqrt(3) * rInvVec[4] * (pscales * thole_q1 + bVec[3])
    else:
        udq_m0 = 0.0
        udq_m1 = 0.0
    ## Uind-Uind
    udud_m0 = -2.0 / 3.0 * rInvVec[3] * (3.0 * (dscales * thole_d0 + bVec[3]) + alphaRVec[3] * X)
    udud_m1 = rInvVec[3] * (dscales * thole_d1 + bVec[3] - 2.0 / 3.0 * alphaRVec[3] * X)
    return cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1


@partial(vmap, in_axes=(0, 0), out_axes=(0))
@jit_condition(static_argnums=())
def get_pair_dmp(pol1, pol2):
    return (pol1*pol2) ** (1/6)

@jit_condition(static_argnums=(2))
def pme_self(Q_h, kappa, lmax=2):
    '''
    This function calculates the PME self energy

    Inputs:
        Q:
            Na * (lmax+1)^2: harmonic multipoles, local or global does not matter
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    '''
    n_harms = (lmax + 1) ** 2
    l_list = np.array([0] + [1,]*3 + [2,]*5)[:n_harms]
    l_fac2 = np.array([1] + [3,]*3 + [15,]*5)[:n_harms]
    factor = kappa/np.sqrt(np.pi) * (2*kappa**2)**l_list / l_fac2
    return - jnp.sum(factor[np.newaxis] * Q_h**2) * DIELECTRIC

# def pme_self(thresh):
#     '''
#     Trim the value at zero point to avoid singularity
#     '''
#     def trim_val_0(x):
#         return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: jnp.array(thresh), lambda x: x])
#     if DO_JIT:
#         return jit(trim_val_0)
#     else:
#         return trim_val_0
def gen_trim_val_0(thresh):
    '''
    Trim the value at zero point to avoid singularity
    '''
    def trim_val_0(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: jnp.array(thresh), lambda x: x])
    if DO_JIT:
        return jit(trim_val_0)
    else:
        return trim_val_0
trim_val_0 = gen_trim_val_0(1e-8)

def gen_trim_val_infty(thresh):
    '''
    Trime the value at infinity to avoid divergence
    '''
    def trim_val_infty(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: x, lambda x: jnp.array(thresh)])
    if DO_JIT:
        return jit(trim_val_infty)
    else:
        return trim_val_infty

trim_val_infty = gen_trim_val_infty(1e8)

@jit_condition(static_argnums=())
def pol_penalty(U_ind, pol):
    '''
    The energy penalty for polarization of each site, currently only supports isotropic polarization:

    Inputs:
        U_ind:
            Na * 3 float: induced dipoles, in isotropic polarization case, cartesian or harmonic does not matter
        pol:
            (Na,) float: polarizability
    '''
    # this is to remove the singularity when pol=0
    pol_pi = trim_val_0(pol)
    # pol_pi = pol/(jnp.exp((-pol+1e-08)*1e10)+1) + 1e-08/(jnp.exp((pol-1e-08)*1e10)+1)
    return jnp.sum(0.5/pol_pi*(U_ind**2).T) * DIELECTRIC


# @partial(vmap, in_axes=(0, 0, None, None), out_axes=0)
@jit_condition(static_argnums=(3))
def calc_e_perm(dr, mscales, kappa, lmax=2):
    r'''
    This function calculates the ePermCoefs at once
    ePermCoefs is basically the interaction tensor between permanent multipole components
    Everything should be done in the so called quasi-internal (qi) frame
    Energy = \sum_ij qiQI * ePermCoeff_ij * qiQJ

    Inputs:
        dr:
            float: distance between one pair of particles
        mscales:
            float: scaling factor between permanent - permanent multipole interactions, for each pair
        kappa:
            float: \kappa in PME, unit in A^-1
        lmax:
            int: max L

    Output:
        cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
            n * 1 array: ePermCoefs
    '''

    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC * (rInv ** i) for i in range(0, 9)])
    alphaRVec = jnp.array([(kappa * dr) ** i for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])

    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i - 1] + (tmp * X / doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

        # C-C: 1
    cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1] * X)
    if lmax >= 1:
        # C-D
        cd = rInvVec[2] * (mscales + bVec[2])
        # D-D: 2
        dd_m0 = -2 / 3 * rInvVec[3] * (3 * (mscales + bVec[3]) + alphaRVec[3] * X)
        dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2 / 3) * alphaRVec[3] * X)
    else:
        cd = 0
        dd_m0 = 0
        dd_m1 = 0

    if lmax >= 2:
        ## C-Q: 1
        cq = (mscales + bVec[3]) * rInvVec[3]
        ## D-Q: 2
        dq_m0 = rInvVec[4] * (3 * (mscales + bVec[3]) + (4 / 3) * alphaRVec[5] * X)
        dq_m1 = -jnp.sqrt(3) * rInvVec[4] * (mscales + bVec[3])
        ## Q-Q
        qq_m0 = rInvVec[5] * (6 * (mscales + bVec[4]) + (4 / 45) * (-3 + 10 * alphaRVec[2]) * alphaRVec[5] * X)
        qq_m1 = - (4 / 15) * rInvVec[5] * (15 * (mscales + bVec[4]) + alphaRVec[5] * X)
        qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4 / 15) * alphaRVec[5] * X)
    else:
        cq = 0
        dq_m0 = 0
        dq_m1 = 0
        qq_m0 = 0
        qq_m1 = 0
        qq_m1 = 0
        qq_m2 = 0

    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2


@partial(vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None), out_axes=0)
@jit_condition(static_argnums=(12, 13))
def pme_real_kernel(dr, qiQI, qiQJ, qiUindI, qiUindJ, thole1, thole2, dmp, mscales, pscales, dscales, kappa, lmax=2, lpol=False):
    '''
    This is the heavy-lifting kernel function to compute the realspace multipolar PME
    Vectorized over interacting pairs

    Input:
        dr:
            float, the interatomic distances, (np) array if vectorized
        qiQI:
            [(lmax+1)^2] float array, the harmonic multipoles of site i in quasi-internal frame
        qiQJ:
            [(lmax+1)^2] float array, the harmonic multipoles of site j in quasi-internal frame
        qiUindI
            (3,) float array, the harmonic dipoles of site i in QI frame
        qiUindJ
            (3,) float array, the harmonic dipoles of site j in QI frame
        thole1
            float: thole damping coeff of site i
        thole2
            float: thole damping coeff of site j
        dmp:
            float: (pol1 * pol2)**1/6, distance rescaling params used in thole damping
        mscale:
            float, scaling factor between interacting sites (permanent-permanent)
        pscale:
            float, scaling factor between perm-ind interaction
        dscale:
            float, scaling factor between ind-ind interaction
        kappa:
            float, kappa in unit A^1
        lmax:
            int, maximum angular momentum
        lpol:
            bool, doing polarization?

    Output:
        energy:
            float, realspace interaction energy between the sites
    '''

    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa, lmax)
    if lpol:
        cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1 = calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax)

    Vij0 = cc*qiQI[0]
    Vji0 = cc*qiQJ[0]
    # C-Uind
    if lpol:
        Vij0 -= cud * qiUindI[0]
        Vji0 += cud * qiUindJ[0]

    if lmax >= 1:
        # C-D
        Vij0 = Vij0 - cd*qiQI[1]
        Vji1 = -cd*qiQJ[0]
        Vij1 = cd*qiQI[0]
        Vji0 = Vji0 + cd*qiQJ[1]
        # D-D m0
        Vij1 += dd_m0 * qiQI[1]
        Vji1 += dd_m0 * qiQJ[1]
        # D-D m1
        Vij2 = dd_m1*qiQI[2]
        Vji2 = dd_m1*qiQJ[2]
        Vij3 = dd_m1*qiQI[3]
        Vji3 = dd_m1*qiQJ[3]
        # D-Uind
        if lpol:
            Vij1 += dud_m0 * qiUindI[0]
            Vji1 += dud_m0 * qiUindJ[0]
            Vij2 += dud_m1 * qiUindI[1]
            Vji2 += dud_m1 * qiUindJ[1]
            Vij3 += dud_m1 * qiUindI[2]
            Vji3 += dud_m1 * qiUindJ[2]

    if lmax >= 2:
        # C-Q
        Vij0 = Vij0 + cq*qiQI[4]
        Vji4 = cq*qiQJ[0]
        Vij4 = cq*qiQI[0]
        Vji0 = Vji0 + cq*qiQJ[4]
        # D-Q m0
        Vij1 += dq_m0*qiQI[4]
        Vji4 += dq_m0*qiQJ[1]
        # Q-D m0
        Vij4 -= dq_m0*qiQI[1]
        Vji1 -= dq_m0*qiQJ[4]
        # D-Q m1
        Vij2 = Vij2 + dq_m1*qiQI[5]
        Vji5 = dq_m1*qiQJ[2]
        Vij3 += dq_m1*qiQI[6]
        Vji6 = dq_m1*qiQJ[3]
        Vij5 = -(dq_m1*qiQI[2])
        Vji2 += -(dq_m1*qiQJ[5])
        Vij6 = -(dq_m1*qiQI[3])
        Vji3 += -(dq_m1*qiQJ[6])
        # Q-Q m0
        Vij4 += qq_m0*qiQI[4]
        Vji4 += qq_m0*qiQJ[4]
        # Q-Q m1
        Vij5 += qq_m1*qiQI[5]
        Vji5 += qq_m1*qiQJ[5]
        Vij6 += qq_m1*qiQI[6]
        Vji6 += qq_m1*qiQJ[6]
        # Q-Q m2
        Vij7  = qq_m2*qiQI[7]
        Vji7  = qq_m2*qiQJ[7]
        Vij8  = qq_m2*qiQI[8]
        Vji8  = qq_m2*qiQJ[8]
        # Q-Uind
        if lpol:
            Vji4 += udq_m0*qiUindJ[0]
            Vij4 -= udq_m0*qiUindI[0]
            Vji5 += udq_m1*qiUindJ[1]
            Vji6 += udq_m1*qiUindJ[2]
            Vij5 -= udq_m1*qiUindI[1]
            Vij6 -= udq_m1*qiUindI[2]

    # Uind - Uind
    if lpol:
        Vij1dd = udud_m0 * qiUindI[0]
        Vji1dd = udud_m0 * qiUindJ[0]
        Vij2dd = udud_m1 * qiUindI[1]
        Vji2dd = udud_m1 * qiUindJ[1]
        Vij3dd = udud_m1 * qiUindI[2]
        Vji3dd = udud_m1 * qiUindJ[2]
        Vijdd = jnp.stack(( Vij1dd, Vij2dd, Vij3dd))
        Vjidd = jnp.stack(( Vji1dd, Vji2dd, Vji3dd))

    if lmax == 0:
        Vij = Vij0
        Vji = Vji0
    elif lmax == 1:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3))
    elif lmax == 2:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
    else:
        raise ValueError(f"Invalid lmax {lmax}. Valid values are 0, 1, 2")

    if lpol:
        return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji)) + jnp.array(0.5) * (jnp.sum(qiUindJ*Vijdd) + jnp.sum(qiUindI*Vjidd))
    else:
        return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji))

# @jit_condition(static_argnums=(7))
def pme_real(positions, box, pairs,
        Q_global, Uind_global, pol, tholes,
        mScales, pScales, dScales,
        kappa, lmax, lpol):
    '''
    This is the real space PME calculate function
    NOTE: only deals with permanent-permanent multipole interactions
    It expands the pairwise parameters, and then invoke pme_real_kernel
    It seems pointless to jit it:
    1. the heavy-lifting kernel function is jitted and vmapped
    2. len(pairs) keeps changing throughout the simulation, the function would just recompile everytime

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 3: interacting pair indices and topology distance
        Q_global:
            Na * (l+1)**2: harmonics multipoles of each atom, in global frame
        Uind_global:
            Na * 3: harmonic induced dipoles, in global frame
        pol:
            (Na,): polarizabilities
        tholes:
            (Na,): thole damping parameters
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        lmax:
            int: maximum L
        lpol:
            Bool: whether do a polarizable calculation?

    Output:
        ene: pme realspace energy
    '''
    # pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    pairs = pairs.at[:, :2].set(regularize_pairs_new(pairs[:, :2]))
    buffer_scales = pair_buffer_scales(pairs[:, :2])
    # buffer_scales = pair_buffer_scales_new(pairs[:, :2])
    box_inv = jnp.linalg.inv(box)
    r1 = distribute_v3(positions, pairs[:, 0])
    r2 = distribute_v3(positions, pairs[:, 1])
    Q_extendi = distribute_multipoles(Q_global, pairs[:, 0])
    Q_extendj = distribute_multipoles(Q_global, pairs[:, 1])
    nbonds = pairs[:, 2]
    #nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    indices = nbonds-1
    mscales = distribute_scalar(mScales, indices)
    mscales = mscales * buffer_scales
    if lpol:
        pol1 = distribute_scalar(pol, pairs[:, 0])
        pol2 = distribute_scalar(pol, pairs[:, 1])
        thole1 = distribute_scalar(tholes, pairs[:, 0])
        thole2 = distribute_scalar(tholes, pairs[:, 1])
        Uind_extendi = distribute_v3(Uind_global, pairs[:, 0])
        Uind_extendj = distribute_v3(Uind_global, pairs[:, 1])
        pscales = distribute_scalar(pScales, indices)
        pscales = pscales * buffer_scales
        dscales = distribute_scalar(dScales, indices)
        dscales = dscales * buffer_scales
        dmp = get_pair_dmp(pol1, pol2)
    else:
        Uind_extendi = None
        Uind_extendj = None
        pscales = None
        dscales = None
        thole1 = None
        thole2 = None
        dmp = None

    # deals with geometries
    dr = r1 - r2
    dr = v_pbc_shift(dr, box, box_inv)
    norm_dr = jnp.linalg.norm(dr, axis=-1)
    Ri = build_quasi_internal(r1, r2, dr, norm_dr)
    qiQI = rot_global2local(Q_extendi, Ri, lmax)
    qiQJ = rot_global2local(Q_extendj, Ri, lmax)
    if lpol:
        qiUindI = rot_ind_global2local(Uind_extendi, Ri)
        qiUindJ = rot_ind_global2local(Uind_extendj, Ri)
    else:
        qiUindI = None
        qiUindJ = None

    # everything should be pair-specific now
    ene = jnp.sum(
        pme_real_kernel(
            norm_dr,
            qiQI,
            qiQJ,
            qiUindI,
            qiUindJ,
            thole1,
            thole2,
            dmp,
            mscales,
            pscales,
            dscales,
            kappa,
            lmax,
            lpol
        ) * buffer_scales
    )

    return ene


# @jit_condition(static_argnums=())
def energy_pme(positions, box, pairs,
        Q_local, Uind_global, pol, tholes,
        mScales, pScales, dScales,
        construct_local_frame_fn, pme_recip_fn, kappa, K1, K2, K3, lmax, lpol, lpme=True):
    '''
    This is the top-level wrapper for multipole PME

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box
        Q_local:
            Na * (lmax+1)^2: harmonic multipoles of each site in local frame
        Uind_global:
            Na * 3: the induced dipole moment, in GLOBAL CARTESIAN!
        pol:
            (Na,) float: the polarizability of each site, unit in A**3
        tholes:
            (Na,) float: the thole damping widths for each atom, it's dimensionless, default is 8 according to MPID paper
        mScales, pScale, dScale:
            (Nexcl,): multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
            for permanent-permanent, permanent-induced, induced-induced interactions
        pairs:
            Np * 3: interacting pair indices and topology distance
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        construct_local_frame_fn:
            function: local frame constructors, from generate_local_frame_constructor
        pme_recip:
            function: see recip.py, a reciprocal space calculator
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        lmax:
            int: maximum L
        lpol:
            bool: if polarizable or not? if yes, 1, otherwise 0
        lpme:
            bool: doing pme? If false, then turn off reciprocal space and set kappa = 0

    Output:
        energy: total pme energy
    '''
    # if doing a multipolar calculation
    if lmax > 0:
        local_frames = construct_local_frame_fn(positions, box)
        Q_global = rot_local2global(Q_local, local_frames, lmax)
    else:
        if lpol:
            # if fixed multipole only contains charge, and it's polarizable, then expand Q matrix
            dips = jnp.zeros((Q_local.shape[0], 3))
            Q_global = jnp.hstack((Q_local, dips))
            lmax = 1
        else:
            Q_global = Q_local

    # note we assume when lpol is True, lmax should be >= 1
    if lpol:
        # convert Uind to global harmonics, in accord with Q_global
        U_ind = C1_c2h.dot(Uind_global.T).T
        Q_global_tot = Q_global.at[:, 1:4].add(U_ind)
    else:
        Q_global_tot = Q_global

    if lpme is False:
        kappa = 0

    if lpol:
        ene_real = pme_real(positions, box, pairs, Q_global, U_ind, pol, tholes,
                           mScales, pScales, dScales, kappa, lmax, True)
    else:
        ene_real = pme_real(positions, box, pairs, Q_global, None, None, None,
                           mScales, None, None, kappa, lmax, False)

    if lpme:
        ene_recip = pme_recip_fn(positions, box, Q_global_tot)
        ene_self = pme_self(Q_global_tot, kappa, lmax)

        if lpol:
            ene_self += pol_penalty(U_ind, pol)
        return ene_real + ene_recip + ene_self

    else:
        if lpol:
            ene_self = pol_penalty(U_ind, pol)
        else:
            ene_self = 0.0
        return ene_real + ene_self

from openmm import *
from dmff import Hamiltonian, NeighborList
if __name__=="__main__":
    pdb = "0.pdb"
    prm = "pme_ff.xml"
    # prm, value",
    # [("tests/data/lj3.pdb", "tests/data/lj3.xml", -2.001220464706421)])
    rcut = 1.0  # nanometers
    pdb = app.PDBFile(pdb)
    h = Hamiltonian(prm)
    box_vectors = np.diag([20, 20, 20]) * unit.nanometer
    pdb.topology.setPeriodicBoxVectors(box_vectors)
    potential = h.createPotential(
        pdb.topology,
        nonbondedMethod=app.PME,
        constraints=app.HBonds,
        removeCMMotion=False,
        rigidWater=False,
        nonbondedCutoff=rcut * unit.nanometers,
        useDispersionCorrection=False,
        PmeCoeffMethod="gromacs",
        PmeSpacing=0.10
    )
    positions = jnp.array(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    )
    box = jnp.array([
        [20.0, 0.00, 0.00],
        [0.00, 20.0, 0.00],
        [0.00, 0.00, 20.0]
    ])

    nbList = NeighborList(box, rcut, potential.meta["cov_map"])
    nbList.allocate(positions)
    pairs = nbList.pairs
    r_cut = 1.0
    map_charge =jnp.zeros(100,int)
    kappa = 3.458910584449768
    K1 = 200
    K2 = 200
    K3 = 200
    top_mat = None
    coulforce = CoulombPMEForce(r_cut, map_charge, kappa,
                                (K1, K2, K3), topology_matrix=top_mat)
    coulenergy = coulforce.generate_get_energy()
    charge = jnp.ones(1,float)
    mscales_coul =jnp.array([0.,0.,1.,1.,1.,1.],float)
    coulE = coulenergy(positions, box, pairs,
                       charge, mscales_coul)

    test_num = 30
    start_time = timeit.default_timer()
    for i in range(test_num):
        coulE = coulenergy(positions, box, pairs,
                           charge, mscales_coul)

    print("计算平均时间 jit :", (timeit.default_timer() - start_time) / test_num)


    print(coulE)

    coulforce1 = CoulombPMEForce_MoveN(r_cut, map_charge, kappa,
                                (K1, K2, K3), topology_matrix=top_mat)
    coulenergy1= coulforce1.generate_get_energy()
    charge = jnp.ones(1, float)
    mscales_coul = jnp.array([0., 0., 1., 1., 1., 1.], float)
    coulE = coulenergy1(positions, box, pairs,
                       charge, mscales_coul)
    start_time = timeit.default_timer()
    for i in range(test_num):
        coulE = coulenergy1(positions, box, pairs,
                           charge, mscales_coul)

    print("计算平均时间_修改N位置 jit :", (timeit.default_timer() - start_time) / test_num)

    print(coulE)
