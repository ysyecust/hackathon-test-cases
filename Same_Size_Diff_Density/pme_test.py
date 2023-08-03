
from openmm import *
from dmff import Hamiltonian, NeighborList
from pme.pme_funcs import *
import openmm
from pme.setting import DO_JIT
if __name__=="__main__":

    DIELECTRIC = 1389.3545584

    DEFAULT_THOLE_WIDTH = 5.0


    pdb = "0.100.pdb"
    prm = "pme_ff.xml"
    # prm, value",
    # [("tests/data/lj3.pdb", "tests/data/lj3.xml", -2.001220464706421)])
    rcut = 1.0  # nanometers
    pdb = openmm.app.PDBFile(pdb)
    h = Hamiltonian(prm)
    box_vectors = np.diag([20, 20, 20]) * unit.nanometer
    pdb.topology.setPeriodicBoxVectors(box_vectors)
    potential = h.createPotential(
        pdb.topology,
        nonbondedMethod=openmm.app.PME,
        constraints=openmm.app.HBonds,
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
    spend_time = (timeit.default_timer() - start_time) / test_num
    if DO_JIT:
        print("计算平均时间 jit:",spend_time)
    # print("计算平均时间:", (timeit.default_timer() - start_time) / test_num)
    else:
        print("计算平均时间:",spend_time)

    print(coulE)
    # with jax.profiler.trace("profile/jax-trace", create_perfetto_link=True):
    # jax.profiler.start_trace("/tmp/tensorboard")
    coulforce1 = CoulombPMEForce_all(r_cut, map_charge, kappa,
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

    spend_time = (timeit.default_timer() - start_time) / test_num
    if DO_JIT:
        print("计算平均时间--jit setup kpt jit:", spend_time)
    # print("计算平均时间:", (timeit.default_timer() - start_time) / test_num)
    else:
        print("计算平均时间--setup kpt:", spend_time)

    # jax.profiler.stop_trace()