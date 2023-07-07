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
            atoms = read(dir / 'init.pdb')


        # allocate resource
        # pairs = nblist.build(xyz)

        # load references results
        # ref_pairs = json.load(open('ref_pairs.json'))

