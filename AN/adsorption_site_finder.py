#!/usr/bin/env python
# coding: utf-8

## ZnDTPとFM剤の酸化鉄表面吸着

import os, sys, csv, time, socket, glob, shutil #, re
from io import StringIO
import pickle
from itertools import combinations, product

# NOTE: Should I set "CUDA_VISIBLE_DEVICES" befor importing modules?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase import Atoms

# from ase.md.langevin import Langevin
from ase.optimize import BFGS, FIRE
from ase import units

# from ase.md.npt import NPT
# from ase.md.verlet import VelocityVerlet
# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import read, write
from ase.io.trajectory import Trajectory

# from ase.io.trajectory import Trajectory
from ase.build import surface, add_adsorbate, bulk
from ase.constraints import FixAtoms, StrainFilter, ExpCellFilter
from ase.visualize import view

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import optuna
from optuna.samplers import TPESampler
import nglview as nv
from nglview.datafiles import PDB, XTC

# from IPython.display import Image, display_png


# %% [markdown]
# ### SMILES文字列から構造(元素と座標)を作成しase.Atomsを返す。

# %%
def smiles_to_atoms(smiles='CC', return_rwmol=False):
    rwmol = Chem.MolFromSmiles(smiles)
    rwmol = Chem.RWMol(rwmol)
    rwmol = AllChem.AddHs(rwmol)
    AllChem.EmbedMolecule(rwmol)
    with StringIO() as f:
        f.write(Chem.MolToPDBBlock(rwmol))
        f.seek(0)
        mol = read(f, format='proteindatabank')
    if return_rwmol:
        return mol, rwmol
    else:
        return mol

# %% [markdown]
# ### 構造可視化の設定

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

# %%



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
# ### 構造最適化

# %%


def myopt(
    atoms,
    fmax=0.05,
    steps=1000,
    maxstep=0.2,
    logfile=None,
    constraintatoms=[],
    calculator=None,
):
    sc = FixAtoms(indices=constraintatoms)
    atoms.set_constraint(sc)
    atoms.set_calculator(calculator)
    opt = BFGS(atoms, maxstep=maxstep, logfile=logfile)
    opt.run(fmax, steps=steps)
    return atoms



# %%

def SetCutoff(a):
    if a.get_pbc()[0] == True or a.get_pbc()[1] == True or a.get_pbc()[2] == True:
        print("Periodic system is not available")
        return 0.0
    else:
        pos = a.get_positions()
        rmin = 1.0e5
        for i in range(len(a)):
            for j in range(i + 1, len(a), 1):
                rx = pos[i, 0] - pos[j, 0]
                ry = pos[i, 1] - pos[j, 1]
                rz = pos[i, 2] - pos[j, 2]
                rr = np.sqrt(rx * rx + ry * ry + rz * rz)
                if rr < rmin:
                    rmin = rr
        rmin *= 0.99
        return rmin


def SetCutoffElem(a):
    if a.get_pbc()[0] == True or a.get_pbc()[1] == True or a.get_pbc()[2] == True:
        print("Periodic system is not available")
        return 0.0
    else:
        pos = a.get_positions()
        sym = a.get_chemical_symbols()
        rminH = 1.0e5
        rmin = 1.0e5
        for i in range(len(a)):
            for j in range(i + 1, len(a), 1):
                rx = pos[i, 0] - pos[j, 0]
                ry = pos[i, 1] - pos[j, 1]
                rz = pos[i, 2] - pos[j, 2]
                rr = np.sqrt(rx * rx + ry * ry + rz * rz)
                if sym[i] != "H" and sym[j] != "H":
                    if rr < rmin:
                        rmin = rr
                else:
                    if rr < rminH:
                        rminH = rr
        rmin *= 0.99
        rminH *= 0.99
        print("Cutoff for hydrogen: {:.2f} A".format(rminH))
        print("Cutoff for others:   {:.2f} A".format(rmin))
        return rmin, rminH



# %% [markdown]
# ### Optunaに渡す関数の定義
# TODO: classにする。

# %%

def init_objective(
    calculator,
    mol,
    slab,
    surface_elements,
    initx=0.0,
    inity=0.0,
    initz=0.0,
    rcut=1.2,
    rcutH=1.0,
    z_fix=6.0,
    fmax=0.1,
    steps=2000,
    maxstep=0.2,
    logfile=None,
    traj=None,
):
    if not hasattr(fmax, "__iter__"):
        fmax = [fmax]

    def _objective(trial):
        xmin = np.min(slab.get_positions()[:, 0])
        xmax = np.max(slab.get_positions()[:, 0])
        ymin = np.min(slab.get_positions()[:, 1])
        ymax = np.max(slab.get_positions()[:, 1])

        x_ang = trial.suggest_uniform("x_ang", 0.0, 360.0)
        y_ang = trial.suggest_uniform("y_ang", 0.0, 360.0)
        z_ang = trial.suggest_uniform("z_ang", 0.0, 360.0)
        x_pos = trial.suggest_uniform("x_pos", xmin, xmax)
        y_pos = trial.suggest_uniform("y_pos", ymin, ymax)
        z_hig = trial.suggest_uniform("z_hig", 2.0, 3.5)

        m = mol.copy()
        m.rotate(initx, "x")
        m.rotate(inity, "y")
        m.rotate(initz, "z")
        m.rotate(x_ang, "x")
        m.rotate(y_ang, "y")
        m.rotate(z_ang, "z")

        b = m.positions[:, 2].argmin()
        ads = slab.copy()
        ads = AddMol(
            ads, m, x_pos, y_pos, z_hig, b, 2.0, surface_elements=surface_elements
        )
        ads.wrap()
        err = CheckOverlapElem(ads, rcut, rcutH)
        if err == 0:
            c = [a.index for a in ads if a.z < z_fix]
            #            ads = myopt (ads, 0.1, 2000, 0.2, None, constraintatoms = c)
            # optimizeの収束条件を徐々に厳しくしつつ、保存や出力をする。
            for _fmax in fmax:
                ads = myopt(
                    ads,
                    fmax=_fmax,
                    steps=steps,
                    maxstep=maxstep,
                    logfile=logfile,
                    constraintatoms=c,
                    calculator=calculator,
                )
                # 後処理 (今は無し)
                
            if traj is not None:
                traj.write(ads)
            return ads.get_potential_energy()
        else:
            return 0.0

    return _objective, traj


# %%


def CheckOverlap(a, rcut):
    lx = a.get_cell_lengths_and_angles()[0]
    ly = a.get_cell_lengths_and_angles()[1]
    lz = a.get_cell_lengths_and_angles()[2]
    pos = a.get_positions()
    err = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a), 1):
            rx = (pos[i, 0] - pos[j, 0]) / lx
            ry = (pos[i, 1] - pos[j, 1]) / ly
            rz = (pos[i, 2] - pos[j, 2]) / lz
            while rx < -0.5:
                rx += 1.0
            while rx > 0.5:
                rx -= 1.0
            while ry < -0.5:
                ry += 1.0
            while ry > 0.5:
                ry -= 1.0
            while rz < -0.5:
                rz += 1.0
            while rz > 0.5:
                rz -= 1.0
            rx *= lx
            ry *= ly
            rz *= lz
            rr = np.sqrt(rx * rx + ry * ry + rz * rz)
            if rr < rcut:
                err += 1
    return err


def CheckOverlapElem(a, rcut, rcutH):
    lx = a.get_cell_lengths_and_angles()[0]
    ly = a.get_cell_lengths_and_angles()[1]
    lz = a.get_cell_lengths_and_angles()[2]
    pos = a.get_positions()
    sym = a.get_chemical_symbols()
    err = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a), 1):
            rx = (pos[i, 0] - pos[j, 0]) / lx
            ry = (pos[i, 1] - pos[j, 1]) / ly
            rz = (pos[i, 2] - pos[j, 2]) / lz
            while rx < -0.5:
                rx += 1.0
            while rx > 0.5:
                rx -= 1.0
            while ry < -0.5:
                ry += 1.0
            while ry > 0.5:
                ry -= 1.0
            while rz < -0.5:
                rz += 1.0
            while rz > 0.5:
                rz -= 1.0
            rx *= lx
            ry *= ly
            rz *= lz
            rr = np.sqrt(rx * rx + ry * ry + rz * rz)

            if sym[i] != "H" and sym[j] != "H":
                if rr < rcut:
                    err += 1
            else:
                if rr < rcutH:
                    err += 1
    return err



# %%

def AddMol(a, m, x, y, h, b, rcut=2.0, surface_elements=["Fe"]):

    if isinstance(surface_elements, str):
        surface_elements = [surface_elements]

    # POSITIONS AND CHEMICAL SYMBOLS
    pa = a.get_positions()
    sa = a.get_chemical_symbols()
    pm = m.get_positions()
    sm = m.get_chemical_symbols()
    # CELL
    lx = a.get_cell_lengths_and_angles()[0]
    ly = a.get_cell_lengths_and_angles()[1]
    lz = a.get_cell_lengths_and_angles()[2]
    # TRANSLATE MOLECULE POSITION
    pm -= pm[b]
    pm += [x, y, 0.0]
    # TOP SURFACE
    shift = 0.1
    z = np.max([atom.z for atom in a if atom.symbol in surface_elements]) - shift
    # CHECK OVERLAP
    rep = True
    while rep is True:
        z += shift
        err = 0
        for i in range(len(a)):
            for j in range(len(m)):
                rx = (pm[j, 0] - pa[i, 0]) / lx
                ry = (pm[j, 1] - pa[i, 1]) / ly
                rz = ((pm[j, 2] + z + h) - pa[i, 2]) / lz
                while rx < -0.5:
                    rx += 1.0
                while rx > 0.5:
                    rx -= 1.0
                while ry < -0.5:
                    ry += 1.0
                while ry > 0.5:
                    ry -= 1.0
                while rz < -0.5:
                    rz += 1.0
                while rz > 0.5:
                    rz -= 1.0
                rx *= lx
                ry *= ly
                rz *= lz
                rr = np.sqrt(rx * rx + ry * ry + rz * rz)
                if rr < rcut:
                    err += 1
        if err == 0:
            rep = False
    # TRANSLATE MOLECULE POSITION
    pm += [0.0, 0.0, z + h]
    # PLACE MOLECULE ON SLAB
    p = np.append(pa, pm, axis=0)
    s = np.append(sa, sm, axis=0)
    return Atoms(symbols=s, positions=p, cell=a.get_cell(), pbc=True)



# %% [markdown]
# ### 分子とスラブの構造をファイルから読み込み、原子をソートする。

# %%

def read_structures(ads_structure_file, slab_structure_file, sort_atoms=True):
    
    if isinstance(ads_structure_file, Atoms):
        mol = ads_structure_file
    elif os.path.exists(ads_structure_file):
        mol = read(ads_structure_file)
    else:
        print(f"adsorbate structure {ads_structure_file} not found")
        print(f"in the current directory {os.getcwd()}")
        sys.exit()

    if isinstance(slab_structure_file, Atoms):
        slab = slab_structure_file
    elif os.path.exists(slab_structure_file):
        slab = read(slab_structure_file)
    else:
        print(f"slab structure {slab_structure_file} not found")
        print(f"in the current directory {os.getcwd()}")
        sys.exit()

    if sort_atoms:
        mol = mol[np.argsort(mol.get_chemical_symbols())]
        slab = slab[np.argsort(slab.get_chemical_symbols())]
    return mol, slab


# %% [markdown]
# ### 構造最適化の関数myoptのラッパー関数。
# 分子、スラブ用に引数、前処理、後処理を加えたもの。

# %%
def optimize_molecule(mol,
        calculator=None,
        fmax=0.1,
        steps=1000,
        maxstep=0.02,
        outfile="mol_opt.json",
        logfile=None,
):

    mol = myopt(
        mol,
        fmax=fmax,
        steps=steps,
        maxstep=maxstep,
        logfile=logfile,
        calculator=calculator,
    )
    E_mol = mol.get_potential_energy()
    print("  E_mol = {:.4f} eV".format(E_mol))
    mol.write(outfile)
    
    return mol, E_mol

def optimize_slab(slab,
                  calculator=None,
                  z_fix=6.0,
                  fmax=0.1,
                  steps=1000000,
                  maxstep=0.02,
                  outfile='slab_opt.json',
                  logfile=None,
                  constraints=None,
                 ):

    if z_fix is not None:
        c = [a.index for a in slab if a.z < z_fix]
    else:
        c = constraints

    slab = myopt(
        slab,
        fmax=fmax,
        steps=steps,
        maxstep=maxstep,
        logfile=logfile,
        constraintatoms=c,
        calculator=calculator,
    )
    E_slab = slab.get_potential_energy()
    print("  E_slab = {:.4f} eV".format(E_slab))
    slab.write(outfile)
    return slab, E_slab




# %%
def check_initial_geometry(mol, 
                           slab,
                          initx=0.0,
                          inity=0.0,
                          initz=0.0,
                          ):

    tiny = sys.float_info.epsilon
    (lx, ly, lz) = slab.get_cell_lengths_and_angles()[0:3]

    m = mol.copy()
    m.rotate(initx, "x")
    m.rotate(inity, "y")
    m.rotate(initz, "z")
    b = m.positions[:, 2].argmin()
    ads = slab.copy()
    add_adsorbate(ads, m, 3.0, (0.5 * lx, 0.5 * ly), mol_index=b)
    d = ads.get_all_distances(mic=True)
    mask_nonzero = np.triu(d) > tiny
    dmin = np.min(d[mask_nonzero])
    print(f"Check the initial geometry: dmin = {dmin:.2f} ")
    return ads, dmin

# %% [markdown]
# ### 吸着構造の探索

# %%

def find_ads_site(mol, 
                  slab, 
                  calculator, 
                  surface_elements=['Fe'], 
                  z_fix=6.0, 
                  # -->  分子の吸着位置の初期値
                  initx=0.0, 
                  inity=0.0, 
                  initz=0.0, 
                  # --> myoptの変数。構造最適化の収束条件
                  fmax=0.1, 
                  steps=1000, 
                  maxstep=0.02, 
                  # --> optunaの変数
                  n_trials=10,  # 探索の回数
                  n_jobs=1,     # 常に１
                  n_startup_sampling=10, # 探索最初のランダムサンプリングの数 
                  n_ei_candidates=24,    # EIの候補の数
                  seed_sampler=None,     # 乱数シード
                  multivariate=False,    # 変数が独立でない場合にTrue 
                  # --> 入出力
                  show_progress_bar=True,
                  trajfile='study.traj',
                  prefix='',
                  logfile=None,
                  group=False,
                  debug=False,
                 ):
    # --> Args for optuna study.
    print("# --> Exploring stable adsorption sites...")

    if trajfile is not None:
        traj = Trajectory(trajfile, "a")
    else:
        traj = None

    rcut, rcutH = SetCutoffElem(mol)
    
    # Optunaのデフォルトのsampler
    # 引数 multivariateを指定し、明示的にcreate_studyに渡す。
    sampler = TPESampler( 
        n_startup_trials=n_startup_sampling,
        n_ei_candidates=n_ei_candidates,
        seed=seed_sampler,
        multivariate=multivariate,
        group=group,
    )
    study = optuna.create_study(sampler=sampler)
    optuna.logging.disable_default_handler()
    #    study.optimize (objective, n_trials = 1000, n_jobs = 1, show_progress_bar = True)
    objective, traj = init_objective(
        calculator,
        mol,
        slab,
        surface_elements,
        initx,
        inity,
        initz,
        rcut,
        rcutH,
        z_fix,
        fmax=fmax,
        steps=steps,
        maxstep=maxstep,
        logfile=logfile,
        traj=traj,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )
    # t1 = time.time()
    # print(f"  Elapsed time : {int(t1 - t0)} sec")
    print(
        f"  Elapsed time : {(study.trials[-1].datetime_complete - study.trials[0].datetime_start).seconds} sec"
    )
    if debug:
        try:
            with open(f"optuna_{study.study_name}.pickle", "wb") as f:
                pickle.dump(study, f)
        except Exception as p:
            study.trials_dataframe().to_csv(f"optuna_{study.study_name}.csv")
    if traj is not None:
        traj.close()

    # 最安定な吸着構造を取得
    x_ang = study.best_params["x_ang"]
    y_ang = study.best_params["y_ang"]
    z_ang = study.best_params["z_ang"]
    x_pos = study.best_params["x_pos"]
    y_pos = study.best_params["y_pos"]
    z_hig = study.best_params["z_hig"]

    m = mol.copy()
    m.rotate(initx, "x")
    m.rotate(inity, "y")
    m.rotate(initz, "z")
    m.rotate(x_ang, "x")
    m.rotate(y_ang, "y")
    m.rotate(z_ang, "z")
    b = m.positions[:, 2].argmin()
    ads = slab.copy()
    ads = AddMol(ads, m, x_pos, y_pos, z_hig, b, 2.0, surface_elements=surface_elements)
    ads.wrap()

    write(prefix + "ads_init.car", ads, format="dmol-car")

    return ads, study, traj


# %%
def save_results(study, E_ads, prefix=''):

    x_ang = study.best_params["x_ang"]
    y_ang = study.best_params["y_ang"]
    z_ang = study.best_params["z_ang"]
    x_pos = study.best_params["x_pos"]
    y_pos = study.best_params["y_pos"]
    z_hig = study.best_params["z_hig"]

    with open(prefix + "ads_optuna.csv", "w") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "x_ang",
                "y_ang",
                "z_ang",
                "x_pos",
                "y_pos",
                "z_hig",
                "ads_eV",
                "ads_kJmol",
            ]
        )
        w.writerow(
            [
                x_ang,
                y_ang,
                z_ang,
                x_pos,
                y_pos,
                z_hig,
                E_ads,
                E_ads * units.mol / units.kJ,
            ]
        )

    ene = [each.value for each in study.trials]
    hyp = [each.params for each in study.trials]
    num = np.argsort(ene)
    with open(prefix + "hyperparam.csv", "w") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "#trial",
                "x_ang",
                "y_ang",
                "z_ang",
                "x_pos",
                "y_pos",
                "z_hig",
                "energy_eV",
            ]
        )
        for i in num:
            w.writerow(
                [
                    i,
                    hyp[i]["x_ang"],
                    hyp[i]["y_ang"],
                    hyp[i]["z_ang"],
                    hyp[i]["x_pos"],
                    hyp[i]["y_pos"],
                    hyp[i]["z_hig"],
                    ene[i],
                ]
            )

# %%
def opt(
    pfp_ver="0102",
    gpu=1,
    prefix=None,
    # --> Args for building structure.
    structure_file="mol.car",
    z_fix=6.0,  # Fix slab atoms at z < z_fix.
    # --> Args for initial and final geometry optimizations.
    fmax=0.1,
    steps=2000,
    maxstep=0.02,
    basename='ads_opt',
    trajfile="opt.traj",
    logfile=None,
    debug=False,
):

    hostname = socket.gethostname()
    now = time.strftime("%Y%m%d %H:%M%S %Z")
    msg = f"{now} {hostname}"
    print(f"\nStarted at {msg}.")
    with open("running", "a") as f:
        print(msg, file=f)

    #    if debug and logfile is None:
    #        basename, expt = os.path.splitext(os.path.basename(__file__))
    #        logfile = f'{basename}.log'
    set_gpu(gpu)

    calculator, estimator, model_builder = setup_pfp_calculator(
        ver=pfp_ver,
        # ver = "0012"
        D_flag=True,
        U_flag=True,
    )

    if prefix is None:
        #        prefix, ext = os.path.splitext(os.path.basename(ads_structure_file))
        prefix = ""
    else:
        prefix = f"{prefix}_"

    # 分子の読み込み
    # 構造最適化済みの吸着構造の読み込み

    atoms = read(structure_file)
    # (lx, ly, lz) = atoms.get_cell_lengths_and_angles()[0:3]
    # df = pd.read_csv(reference_paramfile)

    print("# --> Optimizing the most stable geometry...")
    c = [a.index for a in atoms if a.z < z_fix]
    atoms = myopt(
        atoms,
        fmax=fmax,
        steps=steps,
        maxstep=maxstep,
        trajfile=trajfile,
        logfile=logfile,
        constraintatoms=c,
        calculator=calculator,
    )
    E_pot = atoms.get_potential_energy()
    print("  E_pot = {:.4f} eV".format(E_pot))

    write(prefix + f"{basename}.car", atoms, format="dmol-car")
    atoms.write(f"{basename}.json")

    now = time.strftime("%Y%m%d %H:%M%S %Z")
    msg = f"{now} {hostname}"
    print(f"\nFinished at {msg}.")
    with open("running", "a") as f:
        print(msg, file=f)
    os.rename("running", "finished")



# %%

def run(
    #    ads_structure_file="../kmeans/sarcosine_ole_opt.car",
    #    slab_structure_file="../ads_thick_optuna/NARROW_5x5/ZnDTP/ZnDTP_ads_opt.car",
    # --> Args for building structure.
    ads_structure_file="mol.car",
    slab_structure_file="slab.car",
    surface_elements=["Fe"],
    z_fix=6.0,  # Fix slab atoms at z < z_fix.
    initx=0.0,  # 分子の初期角度
    inity=0.0,  # 分子の初期角度
    initz=0.0,  # 分子の初期角度
    # --> Args for initial and final geometry optimizations.
    initial_opt_fmax=0.1,
    initial_opt_steps=1000,
    initial_opt_maxstep=0.02,
    final_opt_fmax=0.1,
    final_opt_steps=1000000,
    final_opt_maxstep=0.02,
    # --> Args for optuna sampler.
    multivariate=True,
    n_ei_candidates=24,
    n_startup_sampling=10,
    seed_sampler=None,
    # --> Args for optuna study.
    n_trials=10,
    n_jobs=1,
    fmax=0.1,
    steps=1000,
    maxstep=0.02,
    show_progress_bar=True,
    logfile=None,
    trajfile="study.traj",
    #    debug=False,
    # --> Calculator settings.
    # 	prefix = "sarcosine",
    pfp_ver="0102",
    gpu=1,
    prefix=None,
    group=False,
    debug=False,
):


    hostname = socket.gethostname()
    now = time.strftime("%Y%m%d %H:%M%S %Z")
    msg = f"{now} {hostname}"
    print(f"\nStarted at {msg}.")
    with open("running", "a") as f:
        print(msg, file=f)

    #    if debug and logfile is None:
    #        basename, expt = os.path.splitext(os.path.basename(__file__))
    #        logfile = f'{basename}.log'

    if prefix is None:
        #        prefix, ext = os.path.splitext(os.path.basename(ads_structure_file))
        prefix = ""
    else:
        prefix = f"{prefix}_"

    set_gpu(gpu)

    calculator, estimator, model_builder = setup_pfp_calculator(
        ver=pfp_ver,
        # ver = "0012"
        D_flag=True,
        U_flag=True,
    )


    # 分子の読み込み
    # 構造最適化済みの吸着構造の読み込み
    mol, slab = read_structures(
        ads_structure_file, slab_structure_file, sort_atoms=True
    )

    # 分子と吸着構造の構造最適化
    print("# --> Optimizing the molecule and slab geometries...")
    sys.stdout.flush()

    mol, E_mol = optimize_molecule(
        mol,
        fmax=initial_opt_fmax,
        steps=initial_opt_steps,
        maxstep=initial_opt_maxstep,
        logfile=logfile,
        calculator=calculator,
        outfile='mol_opt.json',
    )

    v = surview(mol)
    slab, E_slab = optimize_slab(slab,
                                 z_fix=z_fix,
                                 fmax=initial_opt_fmax,
                                 steps=1e10,
                                 maxstep=initial_opt_maxstep,
                                 logfile=logfile,
                                 calculator=calculator,
                                 outfile='slab_opt.json'
    )

    v = surview(slab)

    ads, dmin = check_initial_geometry(mol, slab, initx=initx, inity=inity, initz=initz)
    v = surview(ads)

    ads, study, traj = find_ads_site(mol, 
                  slab, 
                  z_fix=z_fix, 
                  surface_elements=surface_elements, 
                  calculator=calculator, 
                  initx=initx, 
                  inity=inity, 
                  initz=initz, 
                  n_trials=n_trials,
                  n_jobs=n_jobs,
                  show_progress_bar=show_progress_bar,
                  logfile=None,
                  trajfile="study.traj",
                  fmax=fmax, 
                  steps=steps, 
                  maxstep=maxstep, 
                  n_startup_sampling=n_startup_sampling, 
                  n_ei_candidates=n_ei_candidates, 
                  seed_sampler=seed_sampler, 
                  multivariate=multivariate, 
                  group=group)

    print("# --> Optimizing the most stable geometry...")

    ads, E_mol_slab = optimize_slab(ads,
                  calculator=calculator,
                  z_fix=z_fix,
                  fmax=final_opt_fmax,
                  steps=final_opt_steps,
                  maxstep=final_opt_maxstep,
                  logfile=logfile,
                  outfile='slab_opt.json',
                 )
    print("  E_mol_slab = {:.4f} eV".format(E_mol_slab))

    write(prefix + "ads_opt.car", ads, format="dmol-car")
    ads.write("ads_opt.json")

    E_ads = E_mol_slab - (E_mol + E_slab)
    print("# --> Results ")
    print("Adsorption energy: {:.4f} eV".format(E_ads))
    print("Adsorption energy: {:.4f} kJ/mol".format(E_ads * units.mol / units.kJ))
    v = surview(ads)
   
    save_results(study, E_ads, prefix=prefix)

    now = time.strftime("%Y%m%d %H:%M%S %Z")
    msg = f"{now} {hostname}"
    print(f"\nFinished at {msg}.")
    with open("running", "a") as f:
        print(msg, file=f)
    os.rename("running", "finished")
    return ads, E_ads, study


# %% [markdown]
# ## 吸着させる分子構造の生成
#  SMILES文字列 + RDKit を使って吸着させる分子の構造データ(元素と座標)を生成する。
#  ここでは両端 (SMILESのXとY) それぞれ2種類の官能基の組み合わせ、合計4分子を作成する。

# %%
def test():
    mols, rwmols = test_gen_mols()
    slab, z_fix = test_gen_slab()
    optimized_structures, adsorption_energies, studies = test_opt(mols, slab, z_fix)
    test_view(adsorption_energies, optimized_structures)
    test_analyze()
    clean_files()

# %%

def test_gen_mols():
    smiles_template = 'XCCCC=CCCCY'
    end_fragments_X = ['O', 'c1cc(ccc1)C']
    end_fragments_Y = ['(=O)O', 'N']

    xy = list(product(end_fragments_X, end_fragments_Y))
    smileses = [smiles_template.replace('X', x).replace('Y', y) for x, y in xy]
    print(list(xy))
    print(smileses)

    mols = [smiles_to_atoms(_, return_rwmol=True) for _ in smileses]
    rwmols = [_[1] for _ in mols]
    mols = [_[0] for _ in mols]

    for (_mol, _rwmol) in zip(mols, rwmols):
        display(surview(_mol))
        display(_rwmol)
    return mols, rwmols

# %% [markdown]
# ## 分子を吸着させる表面スラブ模型の作成
# ase.build.bulkをつかってCu(111)表面 6x6x1 4L スラブ構造を作成

# %%
def test_gen_slab():
    bulk_Cu = bulk('Cu', 'fcc')
    repeat = [6, 6, 1]
    num_layers = 4
    vacuum=20.0
    direction = (1, 1, 1)
    slab = surface(bulk_Cu, direction, num_layers, vacuum=vacuum, periodic=True)
    slab = slab * repeat

    # 最下層を固定するための座標z_fixを取得しておく。
    # 表面吸着構造の最適化では、z < z_fix の原子をすべて固定する
    z = np.unique(slab.positions[:, 2])
    print(z)
    z_fix = np.min(z) + 0.5 # 大小判定のために、少し余裕(0.5A)をもたせておく。 
    print(z_fix)
    display(surview(slab))
    return slab, z_fix

# %%
### print関数の出力先をファイルに置き換える
# 以下を実行すると、print関数の結果がノートブックに出力されず、ファイルに保存される。
# --> 出力がjuypterのセルにもどらなかった。loggingをつかうべき。
#print(sys.__stdout__)
#print(sys.stderr)
#print(sys.stderr)

#stdout = open('stdout', 'a')
#stderr = open('stderr', 'a')
#(sys.stdout, stdout) = stdout, sys.stdout
#(sys.stderr, stderr) = stderr, sys.stderr

# %% [markdown]
# ## 吸着構造の最適化
# 4つの分子それぞれについて吸着構造を探索し、結果をファイルに保存する。
# * mol_opt.json　吸着分子の構造 (ASE json)
# * slab_opt.json　スラブの構造 (ASE json)
# * ads_init.car　optunaの探索後の構造 (car)
# * ads_opt.car, ads_opt.json   ads_init.carを最適化した構造 (car, ASE json)
# * ads_optuna.csv  optunaの最適解と最適値
# * hyperparam.csv  optunaの途中経過
# * study.traj　　　optunaで探索された全ての構造(ASE Trajectory)

# %%
def test_opt(mols, slab, z_fix):
    save_file_formats = ['.car', '.csv', '.json', 'traj']
    exist_ok = True  # ファイルがすでに存在する場合に上書きする。
    n_trials=15
    n_startup_sampling=10
    surface_elements = ['Cu']

    optimized_structures = []
    adsorption_energies = []
    studies = []
    for i, mol in enumerate(mols):
        destdir = f'{i}'
        os.makedirs(destdir, exist_ok=exist_ok)
        
        _ads, _e_ads, _study = run(mol, 
                slab, 
                surface_elements,
                multivariate=True,  # 変数が独立でない場合(目視でわかる解をなかなか見つけてくれない場合)にTrueにする。
                z_fix=z_fix, 
                n_startup_sampling=n_startup_sampling,  # ランダムな初期値の数。大きいほど局所解に落ちにくい。
                n_trials=n_trials   # optunaの探索回数。大きいほどより極値に近づく。
                                )
        optimized_structures.append(_ads)
        adsorption_energies.append(_e_ads)
        studies.append(_study)
        
        _files = np.concatenate([glob.glob(f'*{_}') for _ in save_file_formats]).tolist()
        for _file in _files:
            if exist_ok:
                shutil.move(_file, os.path.join(destdir, os.path.basename(_file)))
            else:
                shutil.move(_file, destdir)
    return optimized_structures, adsorption_energies, studies

# %% [markdown]
# 

# %%
## print関数の出力先を元に戻す
#sys.stdout, stdout = stdout, sys.stdout
#sys.stderr, stderr = stderr, sys.stderr
#stdout.close()
#stderr.close()
# 正しく設定できていれば、以下print文の結果が（ファイルではなく）セルに出力される。
# --> jupyter notebookでは動かなかった。
#print(sys.stdout)
#print(sys.stderr)

# %% [markdown]
# ## optunaの探索結果を確認する

# %%
def test_view(adsorption_energies, optimized_structures):
    print(adsorption_energies)
    for _ads in optimized_structures:
        display(surview(_ads))


# %%
def test_analyze(
    srcdir="1",
    trajfile="study.traj",
    adsfile="ads_opt.json",
    paramfile="hyperparam.csv",
    resultfile="ads_optuna.csv",
):

    # --> trajectoryファイルからデータを取得
    traj = read(os.path.join(srcdir, trajfile), index=":")
    ene_trials = np.array([_atoms.get_potential_energy() for _atoms in traj])
    idx_min = ene_trials.argmin()

    print(idx_min)
    # print(ene_trials)
    # --> 最適化後のASE Atomsを取得
    ads = read(os.path.join(srcdir, adsfile))

    # --> csvファイルからデータを取得
    print("Results of optuna:")
    params = pd.read_csv(os.path.join(srcdir, paramfile))
    # 計算に失敗したデータを除く
    params = params[params.energy_eV < 0].sort_values("#trial").reset_index(drop=True)
    idx_min_param = params["energy_eV"].idxmin()
    display(params.loc[[idx_min_param]])
    # display(params)
    print("Adsorption energies of optimized geometries:")
    res = pd.read_csv(os.path.join(srcdir, resultfile))
    display(res)

    plt.plot(ene_trials)
    params.plot(y="energy_eV")

    display(surview(ads))
    display(surview(traj[idx_min]))
    display(view(traj, viewer="ngl"))


# %%
def clean_files():
    removefiles = ['mol_opt.json', 'slab_opt.json', 'ads_init.car', 'study.traj', 'finished', 'running']
    for _file in removefiles:
        if os.path.exists(_file):
            os.remove(_file)

# %%
# test()


