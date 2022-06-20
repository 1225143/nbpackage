# %%
import os
import re
from io import StringIO
from datetime import datetime
from pathlib import Path
from numbers import Integral, Number
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.data import atomic_masses, atomic_numbers
from ase.build import make_supercell

# %%
class Lammps():
    header_tags = ["atoms", "bonds", "angles", "dihedrals", "impropers"]
    header_tags_types = [
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
    ]

    tags_atomic = ["Masses", "Atoms", "Velocities"]
    tags_interactions = ["Bonds", "Angles", "Dihedrals", "Impropers",]
    tags_coeffs = [
        "Pair Coeffs",
        "Nonbond Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ]
    def __init__(self, filename=None):
        self.data = dict()
        if filename is not None:
            self.filename = filename
            self.path = Path(filename).resolve()

    def get_formatter(self, x):
        if issubclass(x, Integral):
            return lambda _:f'{_:6d}'
        elif issubclass(x, Number):
            return lambda _:f'{_:12.8f}'
        else:
            return lambda _:f'{_}'

    def write_data(self, filename, data=None):
        """ alias for self.write_lammpsdata. """
        self.write_lammpsdata(filename, data=data)


    def write_lammpsdata(self, filename, data=None):
        if data is None:
            data = self.data
        now = datetime.now().strftime("%Y%M%d-%H%m%S")
        title = data.get('title') or f'Lammps data file created by nbpackage.AN.ioutils {now}'
        
        header = ''
        for tags in (self.header_tags, self.header_tags_types):
            for _tag in tags:
                if _tag in self.data:
                    _formatter = self.get_formatter(int)
                    header += _formatter(data[_tag]) + f' {_tag}\n'
            header += '\n'
    
        for _x in ('x', 'y', 'z'):
            if f'{_x}lo' in data:
                _tag_low, _tag_high = f'{_x}lo', f'{_x}hi'
                header += f'{data[_tag_low]} {data[_tag_high]} {_tag_low} {_tag_high}\n'
        
        header = header.rstrip("\n") + '\n\n'  # remove extra line breaks.

        body = ''
        for _tag in ('Masses', *self.tags_coeffs, 'Atoms', 'Velocities', *self.tags_interactions):
            if _tag in data:
                body += f'{_tag}\n\n'
                _formatters = [self.get_formatter(_.type) for _ in data[_tag].dtypes]
                body += data[_tag].to_string(index=False, header=False, formatters=_formatters)
                body += '\n\n'
        body = body.rstrip('\n') + '\n'

        lines = f'{title}\n\n{header}{body}'
        if filename is None:
            return lines
        else:
            path = Path(filename)
            path.write_text(lines)
        
    def check_ligpargen_format(self, filename):
        with open(filename) as f:
            lines = f.read()
        return lines

    def unique_data(self, data=None):

        if data is None:
            data = self.data
        
        def _sort(data, key, map): 
                data[key] = data[key].iloc[indx]
                data[key].loc[:, ['type']] = data[key]['type'].apply(lambda x:_map[x])
                data[key] = data[key].sort_values(by='type').reset_index(drop=True)
                return data
     
        if 'Pair Coeffs' in data and 'Masses' in data: 
            _data = pd.merge(data['Pair Coeffs'], data['Masses'], on='type')
            _data = _data[[ 'coeff0', 'coeff1', 'mass']]
            a, indx, inv, count = np.unique(_data.values, return_index=True, return_inverse=True, return_counts=True, axis=0)
            num_types = len(indx)
            if len(_data) > num_types:
                # print(len(_data), len(indx))
                _map = dict(zip(sorted(indx+1), np.arange(1, len(indx)+1)))
                for _key in ('Pair Coeffs', 'Masses'):
                    data = _sort(data, _key, _map)
                _types = [_map[_] for _ in indx[inv]+1]
                data['Atoms'].loc[:, ['type']] = _types
            data['atom types'] = num_types

        _map_header = dict(zip(self.tags_interactions, self.header_tags_types[1:]))
        print(_map_header)
        for _tag_coeffs, _tag_ints in zip(self.tags_coeffs[2:], self.tags_interactions):
            if _tag_coeffs in data:
                print(_tag_coeffs)
                _cols = [_ for _ in data[_tag_coeffs].columns.values if _.lstrip('coeff').isdigit()]
                print(_cols)
                _data = data[_tag_coeffs][_cols]
                print(_data)
                a, indx, inv, count = np.unique(_data.values, return_index=True, return_inverse=True, return_counts=True, axis=0)
                print(indx)
                num_types = len(indx)
                if len(_data) > len(indx):
                    # 
                    print(_tag_coeffs, _tag_ints)
                    print(len(_data), len(indx), _map_header[_tag_ints], num_types)
                    _map = dict(zip(sorted(indx+1), np.arange(1, len(indx)+1)))
                    data = _sort(data, _tag_coeffs, _map)
                    _types = [_map[_] for _ in indx[inv]+1]
                    data[_tag_ints].loc[:, ['type']] = _types
                    print(_tag_ints, num_types)
                    data[_map_header[_tag_ints]] = num_types
            print(_tag_coeffs, data[_tag_coeffs].shape)
        return data

    def read_lammpsdata(self, filename=None, replace_data=True):
    
        data = dict()
        if filename is None:
            lines = self.path.read_text()
        elif isinstance(filename, Path):
            lines = filename.read_text()
            self.path = filename
            self.filename = self.path.as_posix()
        elif os.path.exists(filename):
            self.filename = filename
            self.path = Path(filename).resolve()
            lines = self.path.read_text()
        else:
            lines = filename

        lines += '\n\n' 
        data_blocks, lines = self.parse_block_data(lines)
        data.update(data_blocks)

        data_header, lines = self.parse_header(lines)
        data.update(data_header)
    
        lines = lines.strip()
        data['lines_not_parsed'] = lines

        if replace_data:
            self.data = data
        else:
            self.data.update(data)

        if len(lines)>0:
            print(lines)
    
        return data

    def parse_header(self, lines):
        """ Parse header data in lines. """
        data = dict()
        comments = dict()
        tags = self.header_tags_types + self.header_tags
        
        title, lines = lines.split('\n', 1)
        data['title'] = title

        for _tag in tags:
            p = f'^([-+.\d\s]+)({_tag}.*?)$'
            match = re.split(p, lines, flags=re.MULTILINE)
            if len(match) == 4:
                _prev, _value, _match, _next = match
                _line = _match.split('#')
                if len(_line)>1:
                    comments[_tag] = _line[1]
                data[_tag] = int(_value)
                lines = _prev + _next
        data['header comments'] = comments

        _lines = lines.strip().splitlines()
        lines = []
        for _line in _lines:
            if 'lo' in _line and 'hi' in _line:
                _low, _high, _tag_low, _tag_high = _line.strip().split(None, 3)
                _tag_high = _tag_high.split()[0]
                data[_tag_low] = float(_low)
                data[_tag_high] = float(_high)
            else:
                lines.append(_line)
        others = '\n'.join(lines)

        return data, others

    def parse_block_data(self, lines):
        """ Parse block data begin with tags. """
        
        data = dict()
        tags = self.tags_atomic +  self.tags_interactions +  self.tags_coeffs
        for _tag in tags:
            p = f'(^{_tag}\s*\n\n)(.*?)\n\n'
            match = re.split(p, lines, flags=re.MULTILINE|re.DOTALL)
            if len(match) == 4:
                _prev, _, _match, _next = match
                _lines = re.findall('^.*$', _match, flags=re.MULTILINE)
                data[_tag] = _lines
                lines = _prev + _next
        others = lines    
        data = self._parse_block_data(data)

        return data, others

    def _parse_block_data(self, data):
        tags = self.tags_atomic + self.tags_interactions + self.tags_coeffs
        for _tag, _lines in data.items():
            if _tag in tags:
                _lines = [_line.split('#') for _line in _lines]
                _comments = ['' if len(_line)==1 else _line[1] for _line in _lines]
                _data = [_line[0].split() for _line in _lines]
                if _tag in self.tags_interactions:
                    _columns = ['id', 'type', *[f'i{_}' for _ in range(len(_data[0])-2)]]
                    _data = pd.DataFrame(np.array(_data, dtype=int), columns=_columns)
                    _data['comment'] = _comments
                elif _tag in self.tags_coeffs:
                    _columns = ['type', *[f'coeff{_}' for _ in range(len(_data[0])-1)]]
                    _data = pd.DataFrame(_data, columns=_columns)
                    _ncols = _data.shape[1]
                    for _col in _columns:
                        if _data[_col].apply(lambda x:x.lstrip('-').isdigit()).all():
                            _data[_col] = _data[_col].astype(int)
                        else:
                            _data[_col] = _data[_col].astype(float)
                elif _tag == 'Masses':
                    _data = pd.DataFrame(_data, columns=['type', 'mass'])
                    _data['type'] = _data['type'].astype(int)
                    _data['mass'] = _data['mass'].astype(float)
                    _data['comment'] = _comments
                elif _tag == 'Velocity':
                    _data = pd.DataFrame(np.array(_data, dtype=float), columns=['atomid', 'x', 'y', 'z'])
                    _data['atomid'] = data['atomid'].astype(int)
                    _data['comment'] = _comments
                elif _tag == 'Atoms':
                    _columns = ['id', 'mol-id', 'type', 'q', 'x', 'y', 'z', 'ix', 'iy', 'iz']
                    _int_cols = ['id', 'mol-id', 'type', 'ix', 'iy', 'iz']
                    if len(_data[0]) == 6:
                        _columns.remove('q')
                        _data = pd.DataFrame(_data, columns=_columns[:3] + _columns[4:7], dtype=float)
                    if len(_data[0]) == 7:
                        # atom-tag molecule-tag atom-type q x y z nx ny nz  (nx,ny,nz are optional)
                        _data = pd.DataFrame(_data, columns=_columns[:7], dtype=float)
                    elif len(_data[0])==10:
                        _data = pd.DataFrame(_data, columns=_columns, dtype=float)

                    for _col in _int_cols:
                        if _col in _data:
                            _data[_col] = _data[_col].astype(int)

                data[_tag] = _data
        return data

    def make_supercell(self, p=[5, 5, 5], data=None, reduce_box=False):

        if data is None:
            data = self.data

        if reduce_box:
            pos = data['Atoms'].loc[:, ['x', 'y', 'z']].values
            box_size = np.max(pos, axis=0) - np.min(pos, axis=0)
            data['xhi'] = data['xlo'] + box_size[0]
            data['yhi'] = data['ylo'] + box_size[1]
            data['zhi'] = data['zlo'] + box_size[2]
        
        num_cells = np.prod(p)

        natoms = len(data['Atoms'])
        atoms = self.to_aseAtoms(data=data)
        
        if isinstance(p, Integral):
            supercell = atoms * [p, p, p]
        elif np.array(p).ndim == 1 and len(p)==3:
            supercell = atoms * p
        else:
            supercell = make_supercell(atoms, p)
        
        box = lx, ly, lz, alpha, beta, gamma = supercell.cell.cellpar()
        mass_center = supercell.get_center_of_mass()
        pos = supercell.positions
        mol_ids = data['Atoms'].loc[:, 'mol-id']
        num_mols = mol_ids.values.max() - mol_ids.values.min() + 1
        shift_ids = (np.arange(num_cells) * num_mols).repeat(natoms)
        mol_ids = pd.concat([mol_ids]*num_cells).values + shift_ids

        atom_ids = data['Atoms'].loc[:, 'id']
        shift_ids = (np.arange(num_cells)*natoms).repeat(natoms)
        atom_ids = pd.concat([atom_ids]*num_cells).values + shift_ids

        data['xhi'] = data['xlo'] + lx
        data['yhi'] = data['ylo'] + ly
        data['zhi'] = data['zlo'] + lz
        for _tag in self.header_tags:
            if _tag in data:
                data[_tag] = data[_tag] * num_cells
        
        for _tag in ['Atoms', 'Velocities']:
            if _tag in data:
                data[_tag] = pd.concat([data[_tag]]*num_cells).reset_index(drop=True)
        
        data['Atoms'].loc[:, ['x', 'y', 'z']] = pos
        data['Atoms'].loc[:, 'mol-id'] = mol_ids
        data['Atoms'].loc[:, 'id'] = atom_ids
        for _tag in self.tags_interactions:
            if _tag in data:
                _ndata = len(data[_tag])
                print(_tag, _ndata)
                _cols  = data[_tag].columns.tolist()
                _cols = [_ for _ in _cols if not _ in ['id', 'type', 'comment']]
                shift_ids = (np.arange(num_cells) * natoms).repeat(_ndata)
                shift_ids = np.vstack([shift_ids]*len(_cols)).T
                data[_tag] = pd.concat([data[_tag]]*num_cells).reset_index(drop=True)
                data[_tag].loc[:, _cols] += shift_ids
                shift_ids = (np.arange(num_cells)*_ndata).repeat(_ndata)
                data[_tag].loc[:, 'id'] += shift_ids

        
        new = __class__()
        new.data = data
        
        return new

    def to_aseAtoms(self, data=None):
        if data is None:
            data = self.data
        
        numbers = self._get_numbers_from_masses(data)
        lines = self.write_lammpsdata(filename=None, data=data)
        with StringIO() as f:
            f.write(lines)
            f.seek(0)
            atoms = read(f, format='lammps-data')
        atoms.numbers = numbers
        return atoms

            
    def _get_numbers_from_masses(self, data): 
        _int_masses = np.round(atomic_masses).astype(int)
        mass_number = dict(zip(_int_masses[1:], np.arange(1, len(_int_masses)-1)))
        int_masses = data['Masses']['mass'].values.ravel().round().astype(int)
        type_number = {_type: mass_number.get(_m) for _type, _m in zip(data['Masses']['type'], int_masses)}
        numbers = [type_number.get(_) for _ in data['Atoms']['type'].values]
        return numbers

# %%
pd.DataFrame(np.arange(3)*0.99).values.ravel().round().astype(int)

# %%
pd.DataFrame([np.arange(3)*0.99]).round

# %%
def test():
    l = Lammps('../../ligpargen_data/0.data')
    data = l.read_lammpsdata()
    data = l.unique_data()
    print(data)


