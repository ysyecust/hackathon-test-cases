from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from pathlib import Path
import json

if __name__ == "__main__":
    ans = {}
    # glob all pdb in current folder
    for pdb_path in Path('.').glob('*.pdb'):
        if pdb_path.stem == 'lj':
            continue
        # read PDB file
        pdb = PDBFile(str(pdb_path))
        box_vectors = np.diag([20, 20, 20]) * nanometer
        pdb.topology.setPeriodicBoxVectors(box_vectors)

        forcefield = ForceField('pme_ff.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)


        integrator = VerletIntegrator(0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)

        simulation.context.setPositions(pdb.positions)

        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print(f"Energy: {energy}")

        ans[pdb_path.stem] = energy._value

    json.dump(ans, open('pme.json', 'w'), indent=4)