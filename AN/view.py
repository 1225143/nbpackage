# %% [markdown]
# ## Scripts for visualization

# %%
import nglview as nv
import os
import numpy as np
from ipywidgets import interactive
from ase import Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from pathlib import Path, PosixPath


# %%
def view_molecule(filename, view_type):
    print(view_type)
    if isinstance(filename, str):
        basename, ext = os.path.splitext(filename)
        if ext == '.data':
            format = 'lammps-data'
        else:
            format = ext.lstrip('.')

            mol = read(filename, format=format)
    elif isinstance(filename, Atoms):
        mol = filename
    elif isinstance(filename, PosixPath):
        ext = filename.suffix
        print(ext)
        if ext=='.data':
            format = 'lammps-data'
        else:
            format = str(ext).lstrip('.')
        mol = read(str(filename), format=format)
    else:
        print(filename, type(filename))
    v = view(mol, viewer='ngl')
    v.view.add_representation(view_type)
    print(str(filename))
    display(v)

def view_molecules(interactive, path_files, view_type=['ball+stick', 'distance', 'hyperball', 'label', 'licorice', 'line', 'spacefill', 'surface'], height='600px'):
    interactive_view = interactive(view_molecule, filename=path_files, view_type=view_type)
    output = interactive_view
    output.layout.height = height
    return interactive_view

def view_trajectory(filename, view_type):
    if isinstance(filename, PosixPath):
        filename = str(filename)
    trajectory = Trajectory(filename)
    v = view(trajectory, viewer='ngl')
    v.view.add_representation(view_type)
    print(str(filename))
    display(v)

def view_trajectories(interactive, path_filetraj, view_type=['ball+stick', 'distance', 'hyperball', 'label', 'licorice', 'line', 'spacefill', 'surface'], height='600px'):
    interactive_view = interactive(view_trajectory, filename=path_filetraj, view_type=view_type)
    output = interactive_view.children[-1]
    output.layout.height = height
    return interactive_view



# %%
def test():
    import ase.build
    views = []
    atomses = [ase.build.bulk(_, cubic=True) for _ in ['Cu', 'Al', 'Si']]
    vm = view_molecules(interactive, atomses)
    display(vm)

    traj = Trajectory('tmp.traj', mode='w')
    for a in atomses:
        traj.write(a)
    #_view_traj('tmp.traj', 'ball+stick')
    vt = view_traj(interactive, ['tmp.traj'])
    display(vt)

#test()


# %%


def surview(atoms):
    v = nv.show_ase(atoms, gui=False)
    # v.control.zoom(-0.3)
    v.control.spin([1, 0, 0], 3.141 * 1.4)
    # v.control.spin ([0,1,0], 3.14)
    v.add_representation(
        repr_type="unitcell",
    )
    if len(atoms) > 400:
        v.add_representation(
            repr_type="spacefill",
        )
    v._remote_call("setSize", args=["250px", "400px"])
    v.background = "#161616"
    # add_xyz (v, atoms)
    return v


def add_xyz(view, atoms):
    view.shape.add("text", [0, 0, 0], [0.1, 0.5, 0.1], 4, "   0, 0, 0")
    x = int(sum(atoms.cell[:, 0]))
    y = int(sum(atoms.cell[:, 1]))
    z = int(sum(atoms.cell[:, 2]))
    view.shape.add(
        "text", [x, y, z], [0.1, 0.5, 0.1], 4, str(x) + "," + str(y) + "," + str(z)
    )


def add_no(view, atoms, v=[]):
    if v == []:
        v = list(range(len(atoms)))
        # v = list (range (10)) + list (range (len (atoms) - 20, len (atoms)))
    for i, atom in enumerate(atoms):
        if i in v:
            view.shape.add(
                "text", [atom.x, atom.y, atom.z], [0.1, 0.5, 0.1], 1, " " + str(i)
            )


def add_sym(view, atoms):
    for i, atom in enumerate(atoms):
        view.shape.add(
            "text", [atom.x, atom.y, atom.z], [0.5, 0.5, 0.5], 1, atom.symbol
        )


def add_dis(view, atoms, a1, a2):
    view.shape.add(
        "cylinder",
        [atoms[a1].x, atoms[a1].y, atoms[a1].z],
        [atoms[a2].x, atoms[a2].y, atoms[a2].z],
        [0.9, 0.1, 0.1],
        0.05,
    )
    x = 0.5 * (atoms[a1].x + atoms[a2].x)
    y = 0.5 * (atoms[a1].y + atoms[a2].y)
    z = 0.5 * (atoms[a1].z + atoms[a2].z)
    d = np.round(atoms.get_distance(a1, a2), 2)
    view.shape.add("text", [x, y, z], [0.9, 0.1, 0.1], 3, " " + str(d) + "A")



# %% [markdown]
# 


