import pytest
import numpy as np
import json
from pathlib import Path
from ase.io import read

class TestNblist:

    def test_results(self):
    
        # preset parameters
        r_cutoff = 2.0

        # init neighborlist class
        # nblist = NeighborList(r_cutoff, r_skin, **extra_args)

        # test each case

        for dir in Path('.').iterdir():
            if not dir.stem.istitle():
                continue

            # read pdb
            for pdb in dir.glob('*.pdb'):
                atoms = read(pdb)
                xyz = atoms.arrays['positions']

                # load references results
                # ref_pairs = json.load(open(pdb.parent / f'{pdb.stem}.pair'))
                # ref_cell = json.load(open(pdb.parent / f'{pdb.stem}.cell'))

                # allocate resource
                # pairs = nblist.build(xyz, cell)

                # npt.assert_equal(pairs, ref_pairs)

                # do something that xyz of atoms is changed
                # ....

                # pairs = nblist.update(xyz, cell)
                # npt.assert_equal(pairs, ref_pairs)


