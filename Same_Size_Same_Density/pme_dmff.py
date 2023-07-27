import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from jax import jit, value_and_grad
import jax
# from openmm.app import *
from openmm import *
# from openmm.unit import *
import numpy as np
from pathlib import Path
import json
import timeit
import time
if __name__ =="__main__":
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
    func = potential.getPotentialFunc(names=["NonbondedForce"])
    # func = potential.dmff_potentials["NonbondedForce"]
    ene = func(
        positions,
        box,
        pairs,
        h.paramtree
    )
    func_jit = jax.jit(func)
    ene =func_jit(
        positions,
        box,
        pairs,
        h.paramtree
    )
    # start_time = timeit.default_timer()
    # for i in range(50):
    #     ene = func_jit(
    #     positions,
    #     box,
    #     pairs,
    #     h.paramtree
    #     )
    #
    # print("after jit 计算平均时间:",(timeit.default_timer() - start_time)/50)
    start_time = timeit.default_timer()
    for i in range(100):
        ene = func(
            positions,
            box,
            pairs,
            h.paramtree
        )

    print("before jit 计算平均时间:", (timeit.default_timer() - start_time) / 100)
    # start_time = timeit.default_timer()
    # functionB()
    # print(timeit.default_timer() - start_time)

    # assert np.allclose(ene, value, atol=1e-2)
    print(ene)
    # npt.assert_almost_equal(energy, value, decimal=3)
    #
    # energy = jax.jit(coulE)(pos, box, pairs, h.paramtree)
    # npt.assert_almost_equal(energy, value, decimal=3)



    # pdb_path = "0.pdb"
    # pdb = app.PDBFile(str(pdb_path))
    # box_vectors = np.diag([20, 20, 20]) * unit.nanometer
    # pdb.topology.setPeriodicBoxVectors(box_vectors)
    #
    # forcefield = app.ForceField('pme_ff.xml')
    # system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1 * unit.nanometer,
    #                                  constraints=app.HBonds)
    #
    # integrator = VerletIntegrator(0.002 * unit.picoseconds)
    # simulation = app.Simulation(pdb.topology, system, integrator)
    #
    # simulation.context.setPositions(pdb.positions)
    # # print(pdb.positions)
    # state = simulation.context.getState(getEnergy=True)
    # energy = state.getPotentialEnergy()
    # start_time = timeit.default_timer()
    # for i in range(50):
    #     simulation.context.setPositions(pdb.positions)
    #     energy = state.getPotentialEnergy()
    #
    # print("计算平均时间:", (timeit.default_timer() - start_time) / 50)
    #
    # print(f"Energy: {energy}")

    # ans[pdb_path.stem] = energy._value

    # npt.assert_almost_equal(energy, value, decimal=3)
    #
    # energy = jax.jit(ljE)(pos, box, pairs, h.paramtree)
    # npt.assert_almost_equal(energy, value, decimal=3)