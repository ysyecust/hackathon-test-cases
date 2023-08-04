import jax.numpy as jnp
import jax
DIELECTRIC =1389.3545584
import pickle
import timeit
pme_order = 6
# @jax.vmap
def bspline(u, order=pme_order):
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

def pme_recip_canjit1(N,Q_mesh,positions,box,Q):
    # N = jnp.array(N)
    pme_order = 6
    # global variables for the reciprocal module, all related to pme_order
    bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    n_mesh = pme_order ** 3
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
    # N = np.array([K1, K2, K3])
    # Q_mesh = spread_Q(positions, box, Q)
    #--------spread_Q 函数展开--------
    # Nj_Aji_star = get_recip_vectors(N, box)-------------->
    Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
    # For each atom, find the reference mesh point, and u position of the site
    # m_u0, u0 = u_reference(positions, Nj_Aji_star)----------->
    R_in_m_basis = jnp.einsum("ij,kj->ki", Nj_Aji_star, positions)
    m_u0 = jnp.ceil(R_in_m_basis).astype(int)
    u0 = (m_u0 - R_in_m_basis) + pme_order / 2


    # find out the STGO values of each grid point
    # sph_harms = sph_harmonics_GO(u0, Nj_Aji_star)---------->
    # n_harm = ((lmax + 1) ** 2).astype(int)
    # n_harm = ((lmax + 1) ** 2)
    n_harm = 1
    N_a = u0.shape[0]
    # mesh points around each site
    u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a * n_mesh, 3))

    # M_u = bspline(u)
    #M_u = bspline(u) ---------------->


    M_u = bspline(u)
    # theta = theta_eval(u, M_u)-------->
    theta = jnp.prod(M_u, axis=-1)
    sph_harms = theta.reshape(N_a, n_mesh, n_harm)

    # find out the local meshed values for each site
    # Q_mesh_pera = Q_m_peratom(Q, sph_harms)------------>
    N_a = sph_harms.shape[0]
    Q_dbf = Q[:, 0:1]
    Q_mesh_pera = jnp.sum(Q_dbf[:, jnp.newaxis, :] * sph_harms, axis=2)
    # Q_mesh =  Q_mesh_on_m(Q_mesh_pera, m_u0, N)---------->
    indices_arr = jnp.mod(m_u0[:, jnp.newaxis, :] + shifts, N[jnp.newaxis, jnp.newaxis, :])
    ### jax trick implementation without using for loop
    ### NOTICE: this implementation does not work with numpy!
    # Q_mesh = jnp.zeros((N[0], N[1], N[2]))
    # print("Q_mesh:",Q_mesh.shape)
    # print("indices_arr:",indices_arr.sahpe)
    Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)



    #--------spread_Q 函数展开--------
    # N = N.reshape(1, 1, 3)
    # kpts_int = setup_kpts_integer(N)------------->
    # N_half = N.reshape(3)
    # kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2), - (N_half[i] - 1) // 2) for
    #               i in range(3)]
    # kpts_int = jnp.hstack([ki.flatten()[:, jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
    return Q_mesh
    # kpts_int = setup_kpts_integer(N)
def data_create(pme_order):
    bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    n_mesh = pme_order ** 3
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
    return shifts
data_create_jit = jax.jit(data_create)
def pme_recip_part1(N,Q_mesh,positions,box,Q,pme_order,shifts):
    # N = jnp.array(N)
    # pme_order = 6
    # # global variables for the reciprocal module, all related to pme_order
    # # bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    n_mesh = pme_order ** 3
    # shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
    # N = np.array([K1, K2, K3])
    # Q_mesh = spread_Q(positions, box, Q)
    #--------spread_Q 函数展开--------
    # Nj_Aji_star = get_recip_vectors(N, box)-------------->
    Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
    # For each atom, find the reference mesh point, and u position of the site
    # m_u0, u0 = u_reference(positions, Nj_Aji_star)----------->
    R_in_m_basis = jnp.einsum("ij,kj->ki", Nj_Aji_star, positions)
    m_u0 = jnp.ceil(R_in_m_basis).astype(int)
    u0 = (m_u0 - R_in_m_basis) + pme_order / 2


    # find out the STGO values of each grid point
    # sph_harms = sph_harmonics_GO(u0, Nj_Aji_star)---------->
    # n_harm = ((lmax + 1) ** 2).astype(int)
    # n_harm = ((lmax + 1) ** 2)
    n_harm = 1
    N_a = u0.shape[0]
    # mesh points around each site
    u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a * n_mesh, 3))

    # M_u = bspline(u)
    #M_u = bspline(u) ---------------->


    M_u = bspline(u)
    # theta = theta_eval(u, M_u)-------->
    theta = jnp.prod(M_u, axis=-1)
    sph_harms = theta.reshape(N_a, n_mesh, n_harm)

    # find out the local meshed values for each site
    # Q_mesh_pera = Q_m_peratom(Q, sph_harms)------------>
    N_a = sph_harms.shape[0]
    Q_dbf = Q[:, 0:1]
    Q_mesh_pera = jnp.sum(Q_dbf[:, jnp.newaxis, :] * sph_harms, axis=2)
    # Q_mesh =  Q_mesh_on_m(Q_mesh_pera, m_u0, N)---------->
    indices_arr = jnp.mod(m_u0[:, jnp.newaxis, :] + shifts, N[jnp.newaxis, jnp.newaxis, :])
    ### jax trick implementation without using for loop
    ### NOTICE: this implementation does not work with numpy!
    # Q_mesh = jnp.zeros((N[0], N[1], N[2]))
    # print("Q_mesh:",Q_mesh.shape)
    # print("indices_arr:",indices_arr.sahpe)
    Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)



    #--------spread_Q 函数展开--------
    # N = N.reshape(1, 1, 3)
    # kpts_int = setup_kpts_integer(N)------------->
    # N_half = N.reshape(3)
    # kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2), - (N_half[i] - 1) // 2) for
    #               i in range(3)]
    # kpts_int = jnp.hstack([ki.flatten()[:, jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
    return Q_mesh
def box_inv_f(box):
    return jnp.linalg.inv(box)
def kpts_f(box_inv):
    kpts = 2 * jnp.pi * kpts_int.dot(box_inv)
    ksq = jnp.sum(kpts ** 2, axis=1)
    # 4 * K
    # kpts = jnp.hstack((kpts, ksq[:, jnp.newaxis])).T
    kpts = jnp.concatenate((kpts.T, ksq[None, :]), axis=0)
    return kpts,ksq
def m_f(pme_order):
    m = jnp.linspace(-pme_order // 2 + 1, pme_order // 2 - 1, pme_order - 1).reshape(pme_order - 1, 1, 1)
    return m
def theta_f(m,pme_order,kpts_int,N):
    theta_k = jnp.prod(
        jnp.sum(
            bspline(m + pme_order / 2) * jnp.cos(2 * jnp.pi * m * kpts_int[jnp.newaxis] / N),
            axis=0
        ),
        axis=1
    )
    return theta_k
def theta_f1(m,pme_order,kpts_int,N):
    m_pme_order = m + pme_order / 2
    m_pme_order_kpts_int_N = m_pme_order * kpts_int[jnp.newaxis] / N
    cos_term = jnp.cos(2 * jnp.pi * m_pme_order_kpts_int_N)
    bspline_term =bspline(m_pme_order)
    sum_term = jnp.sum(bspline_term * cos_term, axis=0)
    # sum_term = jax.lax.reduce(bspline_term * cos_term, axis=0)
    theta_k = jnp.prod(sum_term, axis=1)
    # theta_k = jax.lax.reduce_prod(sum_term, axis=1)

    return theta_k
def theta_f2(m,pme_order,kpts_int,N):
    theta_k = theta_f1_jit(m,pme_order,kpts_int,N)
    return theta_k
def SK_f(box,Q_mesh):
    V = jnp.linalg.det(box)
    S_k = jnp.fft.fftn(Q_mesh).flatten()
    return V,S_k
def gamma_f(gamma,kappa,V,kpts,S_k,theta_k):
    if not gamma:
        # C_k = Ck_fn(kpts[3, 1:], kappa, V)
        C_k = 2 * jnp.pi / V / kpts[3, 1:] * jnp.exp(-kpts[3, 1:] / 4 / kappa ** 2)
        E_k = C_k * jnp.abs(S_k[1:] / theta_k[1:]) ** 2
        return jnp.sum(E_k) * DIELECTRIC
    else:
        # C_k = Ck_fn(kpts[3, :], kappa, V)
        C_k = 2 * jnp.pi / V / kpts[3, :] * jnp.exp(-kpts[3, :] / 4 / kappa ** 2)
        # debug
        # for i in range(1000):
        #     print('%15.8f%15.8f'%(jnp.real(C_k[i]), jnp.imag(C_k[i])))
        E_k = C_k * jnp.abs(S_k / theta_k) ** 2
        return jnp.sum(E_k)
box_inv_f_jit = jax.jit(box_inv_f)
kpts_f_jit = jax.jit(kpts_f)
m_f_jit = jax.jit(m_f,static_argnums=(0))
theta_f1_jit = jax.jit(theta_f1)
SK_f_jit = jax.jit(SK_f)
gamma_f_jit = jax.jit(gamma_f,static_argnums=(0))
def pme_recip_2(N,box,kpts_int,Q_mesh,gamma,kappa,pme_order):
    box_inv = box_inv_f(box)
    kpts,ksq = kpts_f(box_inv)
    m = m_f(pme_order)
    theta_k = theta_f1(m,pme_order,kpts_int,N)
    # theta_k = theta_f2(m,pme_order,kpts_int,N)
    V,S_k = SK_f(box,Q_mesh)
    result = gamma_f(gamma, kappa, V, kpts, S_k, theta_k)
    return result
def pme_recip_2_j(N,box,kpts_int,Q_mesh,gamma,kappa,pme_order):
    # box_inv = box_inv_f_jit(box)
    # kpts,ksq = kpts_f_jit(box_inv)
    # m = m_f_jit(pme_order)
    # theta_k = theta_f1_jit(m,pme_order,kpts_int,N)
    # V,S_k = SK_f_jit(box,Q_mesh)
    # result = gamma_f_jit(gamma, kappa, V, kpts, S_k, theta_k)
    box_inv = box_inv_f(box)
    kpts,ksq = kpts_f(box_inv)
    m = m_f(pme_order)
    theta_k = theta_f2(m,pme_order,kpts_int,N)
    V,S_k = SK_f(box,Q_mesh)
    result = gamma_f(gamma, kappa, V, kpts, S_k, theta_k)
    return result

def pme_recip_canjit2(N,box,kpts_int,Q_mesh,gamma,kappa):
    # kpts = setup_kpts(box, kpts_int)----------->
    box_inv = jnp.linalg.inv(box)
    # K * 3, coordinate in reciprocal space
    kpts = 2 * jnp.pi * kpts_int.dot(box_inv)
    ksq = jnp.sum(kpts ** 2, axis=1)
    # 4 * K
    # kpts = jnp.hstack((kpts, ksq[:, jnp.newaxis])).T
    kpts = jnp.concatenate((kpts.T, ksq[None, :]), axis=0)
    pme_order= 6
    m = jnp.linspace(-pme_order // 2 + 1, pme_order // 2 - 1, pme_order - 1).reshape(pme_order - 1, 1, 1)
    # m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
    theta_k = jnp.prod(
        jnp.sum(
            bspline(m + pme_order / 2) * jnp.cos(2 * jnp.pi * m * kpts_int[jnp.newaxis] / N),
            axis=0
        ),
        axis=1
    )
    V = jnp.linalg.det(box)
    S_k = jnp.fft.fftn(Q_mesh).flatten()
    # for electrostatic, need to exclude gamma point
    # for dispersion, need to include gamma point
    # return jax.lax.cond(not gamma,calc_true,calc_false,(kappa, V, kpts, S_k, theta_k))
    # return  jax.lax.cond(gamma,lambda  :
    #                      jnp.sum((2 * jnp.pi / V / kpts[3, :] * jnp.exp(-kpts[3, :] / 4 / kappa ** 2))*jnp.abs(S_k / theta_k) ** 2),
    #                      lambda  :
    #                      jnp.sum((2 * jnp.pi / V / kpts[3, 1:] * jnp.exp(-kpts[3, 1:] / 4 / kappa ** 2))*jnp.abs(S_k[1:] / theta_k[1:]) ** 2)*DIELECTRIC,
    #                      )
    if not gamma:
        # C_k = Ck_fn(kpts[3, 1:], kappa, V)
        C_k = 2 * jnp.pi / V / kpts[3, 1:] * jnp.exp(-kpts[3, 1:] / 4 / kappa ** 2)
        E_k = C_k * jnp.abs(S_k[1:] / theta_k[1:]) ** 2
        return jnp.sum(E_k) * DIELECTRIC
    else:
        # C_k = Ck_fn(kpts[3, :], kappa, V)
        C_k = 2 * jnp.pi / V / kpts[3, :] * jnp.exp(-kpts[3, :] / 4 / kappa ** 2)
        # debug
        # for i in range(1000):
        #     print('%15.8f%15.8f'%(jnp.real(C_k[i]), jnp.imag(C_k[i])))
        E_k = C_k * jnp.abs(S_k / theta_k) ** 2
        return jnp.sum(E_k)
def setup_kpts_integer(N):
    """
    Outputs:
        kpts_int:
            n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
    """
    N = jnp.array(N)
    N_half = N.reshape(3)
    kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2), - (N_half[i] - 1) // 2) for
                  i in range(3)]
    kpts_int = jnp.hstack([ki.flatten()[:, jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
    return kpts_int
# pme_recip_canjit1 = jax.jit(pme_recip_canjit1,static_argnums=(0))
pme_recip_canjit1 = jax.jit(pme_recip_canjit1)
pme_recip_canjit2 = jax.jit(pme_recip_canjit2,static_argnums=(4))
# pme_recip_canjit2 = jit(pme_recip_canjit2)
if __name__=="__main__":
    f = open('positions.pckl','rb')
    positions =pickle.load(f)
    f.close()
    N = jnp.array([[[200,200,200]]])
    box = jnp.eye(3)*200
    Q_mesh = jnp.zeros((200,200,200))
    # N1 = tuple([200,200,200])
    N1 = jnp.array([200,200,200])
    Q_global_tot = jnp.ones(4188).reshape(4188,1)
    bspline_range = jnp.arange(-pme_order // 2, pme_order // 2)
    start_time = timeit.default_timer()
    test_num = 20
    pme_order = 6
    for i in range(test_num):
        shifts = data_create(pme_order=pme_order)
        Q_mesh =  pme_recip_part1(N = N1,Q_mesh = Q_mesh,positions=positions,box=box,Q=Q_global_tot,pme_order=pme_order,shifts=shifts)
        # Q_mesh = pme_recip_canjit1(N=N1, Q_mesh=Q_mesh, positions=positions, box=box, Q=Q_global_tot)
        kpts_int = setup_kpts_integer(N)
    spend_time = (timeit.default_timer() - start_time) / test_num
    print("pme1 计算平均时间:", spend_time)
    ene_recip = pme_recip_2_j(N, box, kpts_int, Q_mesh, False, 0.3458910584449768, 6)
    start_time = timeit.default_timer()
    for i in range(test_num):
        ene_recip = pme_recip_2(N, box, kpts_int, Q_mesh, False, 0.3458910584449768,6)

    spend_time = (timeit.default_timer() - start_time) / test_num
    print("pme2  theta 计算平均时间:",spend_time)
    start_time = timeit.default_timer()
    for i in range(test_num):
        ene_recip = pme_recip_2_j(N, box, kpts_int, Q_mesh, False, 0.3458910584449768, 6)

    spend_time = (timeit.default_timer() - start_time) / test_num
    print("pme2 jit theta 计算平均时间:", spend_time)


