# # 本节所需所有package
import asap3
import numpy as np
# import json
# from typing import List
import ase
from ase.io import read, write, Trajectory
from pathlib import Path

if __name__ == "__main__":
    # glob all pdb in current folder
    for pdb in Path('.').glob('*.pdb'):
        if pdb.stem == 'lj':
            continue
        # read pdb
        atoms = read(pdb)
        # get the name of pdb
        name = pdb.stem
        # get the positions of atoms
        xyz = atoms.arrays['positions']
        # set the size of cell
        size = float(name.split('_')[1])
        atoms.set_cell(np.diag([size, size, size]))
        atoms.set_pbc([True, True, True])
        
        # get_neighborlist
        cutoff = 2.0  # cutoff radius      
        nl = asap3.FullNeighborList(cutoff, atoms)
        pair_i_idx = []
        pair_j_idx = []
        n_diff = []
        for i in range(len(atoms)):
            indices, diff, _ = nl.get_neighbors(i)
            pair_i_idx += [i] * len(indices)               # local index of pair i
            pair_j_idx.append(indices)   # local index of pair j
            n_diff.append(diff)

        pair_j_idx = np.concatenate(pair_j_idx)
        pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
        n_diff = np.concatenate(n_diff)
        
        # save answer as txt
        np.savetxt(f'{name}.pair', pairs, fmt='%d')
        # save cell as txt
        np.savetxt(f'{name}.cell', atoms.cell, fmt='%f')        
