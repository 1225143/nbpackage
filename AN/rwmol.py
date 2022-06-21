# %% [markdown]
# ## Scripts for read/write molecules.

# %%
from io import StringIO
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.io import read 

# %%
def get_rwmol_from_smiles(smiles, add_Hs=True, add_conformer=True):
    is_scalar=False
    rwmols = []
    if hasattr(smiles, 'isalpha'):
        is_scalar=True
        smiles = [smiles]
    for _smiles in smiles:
        mol = Chem.MolFromSmiles(_smiles)
        if add_Hs:
            mol = AllChem.AddHs(mol)
        if add_conformer:
            AllChem.EmbedMolecule(mol)
        rwmol = AllChem.RWMol(mol)
        if is_scalar:
            return rwmol
        else:
            rwmols.append(rwmol)     
    return rwmols
 

# %%
def rwmol_to_aseAtoms(rwmol):
    lines = Chem.MolToPDBBlock(rwmol)
    with StringIO() as f:
        f.write(lines)
        f.seek(0)
        a = read(f, format='proteindatabank')
    return a 

# %%
def test():
    s = "CCc1ccc(cc1)"
    m = get_rwmol_from_smiles(s)
    print(m)

    a = rwmol_to_aseAtoms(m)
    print(a)

    

# %%


# %%


# %% [markdown]
# 

