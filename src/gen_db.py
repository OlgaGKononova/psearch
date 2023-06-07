#!/usr/bin/env python3
# ==============================================================================

import os
import sys
import time

from pmapper import utils
from multiprocessing import Pool, cpu_count

from src.database import DB


def molecules_iterator(rdk_molecules, bin_step, pharm_def):
    box_mol_names = set()
    box_mols = set()

    for mol_name, isomers in rdk_molecules: 
        if mol_name in box_mol_names:
            sys.stderr.write(f'\nThe molecule name {mol_name} meets the second time and will be omitted\n')
        elif any(mol in box_mols for i, mol in isomers.items()):
            for i, mol in isomers.items():
                if mol in box_mols:
                    sys.stderr.write(f'\nThe isomer {i} meets the second time and will be omitted\n')
        else:
            box_mol_names.add(mol_name)
            for i, mol in isomers.items():
                box_mols.add(mol)
        yield isomers, mol_name, bin_step, pharm_def


def map_genenerate_pharm_features(args):
    return generate_pharm_features(*args)


def generate_pharm_features(isomers, mol_name, bin_step, pharm_def=None):
    mol_dict, ph_dict, fp_dict = dict(), dict(), dict()

    for i, mol in isomers.items():
        #phs = utils.load_multi_conf_mol(mol, factory=ChemicalFeatures.BuildFeatureFactory(pharm_def), bin_step=bin_step)
        phs = utils.load_multi_conf_mol(mol, smarts_features=None, bin_step=bin_step)
        mol_dict[i] = mol
        ph_dict[i] = [ph.get_feature_coords() for ph in phs]
        fp_dict[i] = [ph.get_fp() for ph in phs]
    return mol_name, mol_dict, ph_dict, fp_dict


def create_conformations_db(rdk_molecules, ncpu, bin_step=1, db_filename=None, pharm_def=None, verbose=True):
    if verbose:
        sys.stderr.write('Database creation started\n')

    start_time = time.time()

    db = DB(filename=db_filename)
    db.write_bin_step(bin_step)

    nprocess = min(cpu_count(), max(ncpu, 1))
    p = Pool(nprocess)

    try:
        for i, (mol_name, mol_dict, ph_dict, fp_dict) in enumerate(
                p.imap_unordered(map_genenerate_pharm_features, molecules_iterator(rdk_molecules, bin_step, pharm_def), chunksize=1), 1):
            
            db.write_mol(mol_name, mol_dict)
            db.write_pharm(mol_name, ph_dict)
            db.write_fp(mol_name, fp_dict)

            if verbose and i % 10 == 0:
                current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                sys.stderr.write('\r{} molecules passed/conformers {}'.format(i, current_time))
                sys.stderr.flush()

    finally:
        p.close()

    if verbose:
        sys.stderr.write("\n")
        
    return db
