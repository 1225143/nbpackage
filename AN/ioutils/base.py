# %%
import os
import sys
import numpy as np
from ase.io import read


# %%
def read_structures(ads_structure_file, slab_structure_file, sort_atoms=True):

    if not os.path.exists(ads_structure_file):
        print(f"adsorbate structure {ads_structure_file} not found")
        print(f"in the current directory {os.getcwd()}")
        sys.exit()
    elif not os.path.exists(slab_structure_file):
        print(f"slab structure {slab_structure_file} not found")
        print(f"in the current directory {os.getcwd()}")
        sys.exit()
    mol = read(ads_structure_file)
    slab = read(slab_structure_file)
    if sort_atoms:
        mol = mol[np.argsort(mol.get_chemical_symbols())]
        slab = slab[np.argsort(slab.get_chemical_symbols())]
    return mol, slab

# %%



