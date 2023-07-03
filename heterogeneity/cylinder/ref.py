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
        x_max, y_max, z_max = np.max(xyz, axis=0)
        x_min, y_min, z_min = np.min(xyz, axis=0)
        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min
        atoms.set_cell(np.diag([20, 20, 20]))
        atoms.set_pbc([x_len, y_len, z_len])
        
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