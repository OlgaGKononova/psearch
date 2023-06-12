#!/usr/bin/env python3
# author          : Pavel Polishchuk
# date            : 04.03.20
# license         : BSD-3
# ==============================================================================

__author__ = 'Pavel Polishchuk'

import os
import shelve


class DB:
    def __init__(self, db_name, filename=None, flag="n", input_data={}):
        self._name = db_name
        if filename:
            self._db = shelve.open(filename=filename, flag=flag, protocol=4)
        else:
            self._db = shelve.Shelf(input_data)

    @property
    def name(self):
        return self._name
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._db.close()
        
    def read(self, fname, flag='c'):
        self._db = shelve.open(os.path.splitext(fname)[0], flag=flag, protocol=4)
        
    def save(self, fname):
        buffer = shelve.open(fname, flag="n", protocol=4)
        for k, v in self._db.items():
            buffer[k] = v
        buffer.sync()
        buffer.close()

    def write_bin_step(self, bin_step):
        self._db['_bin_step'] = bin_step

    def write_mol(self, mol_name, mol_dict):
        self._db[f'{mol_name}_mol'] = mol_dict

    def write_pharm(self, mol_name, pharm_dict):
        self._db[f'{mol_name}_pharm'] = pharm_dict

    def write_fp(self, mol_name, fp_dict):
        self._db[f'{mol_name}_fp'] = fp_dict

    def get_bin_step(self):
        return self._db['_bin_step']

    def get_mol(self, mol_name):
        return self._db[f'{mol_name}_mol']

    def get_pharm(self, mol_name):
        return self._db[f'{mol_name}_pharm']

    def get_fp(self, mol_name):
        return self._db[f'{mol_name}_fp']

    def get_mol_names(self):
        names = list(self._db.keys())
        names.remove('_bin_step')
        names = [n[:-4] for n in names if n.endswith('_mol')]
        return tuple(names)

    def get_conf_count(self, mol_name):
        fps = self.get_fp(mol_name)
        return sum(len(item) for item in fps.values())
