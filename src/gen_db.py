#!/usr/bin/env python3
# author          : Pavel Polishchuk
# date            : 03.04.20
# license         : BSD-3
# ==============================================================================

__author__ = 'Pavel Polishchuk'

import os
import sys
import time
import gzip
import pickle
import argparse
import pandas as pd
from pmapper import utils
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from psearch.scripts.read_input import read_input
from psearch.database import DB


def prep_input(fname, nconf, nstereo, energy, rms, seed, bin_step, pharm_def):
    box_mol_names = set()
    box_mols = set()
    for mol, mol_name in read_input(fname):
        if mol_name in box_mol_names:
            sys.stderr.write(f'\nThe molecule name {mol_name} meets the second time and will be omitted\n')
        elif mol in box_mols:
            sys.stderr.write(f'\nThe molecule {mol} meets the second time and will be omitted\n')
        else:
            box_mol_names.add(mol_name)
            box_mols.add(mol)
        yield mol, mol_name, nconf, nstereo, energy, rms, seed, bin_step, pharm_def


def map_gen_data(args):
    return gen_data(*args)


def gen_stereo(mol, num_isomers):
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
    opts = StereoEnumerationOptions(tryEmbedding=True, maxIsomers=num_isomers)
    isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    return isomers


def gen_conf(mol, num_confs, seed):
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, maxAttempts=num_confs*4, randomSeed=seed)
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    return mol


def remove_confs(mol, energy, rms):
    e = []
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
        if ff is None:
            print(Chem.MolToSmiles(mol))
            return
        e.append((conf.GetId(), ff.CalcEnergy()))
    e = sorted(e, key=lambda x: x[1])

    if not e:
        return

    kept_ids = [e[0][0]]
    remove_ids = []
    
    if energy is not None:
        for item in e[1:]:
            if item[1] - e[0][1] <= energy:
                kept_ids.append(item[0])
            else:
                remove_ids.append(item[0])

    if rms is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(kept_ids, 2)]
        while any(item[2] < rms for item in rms_list):
            for item in rms_list:
                if item[2] < rms:
                    remove_ids.append(item[1])
                    rms_list = [i for i in rms_list if i[0] != item[1] and i[1] != item[1]]
                    break

    for cid in set(remove_ids):
        mol.RemoveConformer(cid)

    # conformers are reindexed staring with 0 step 1
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)


def gen_data(mol, mol_name, nconf, nstereo, energy, rms, seed, bin_step, pharm_def):
    mol_dict, ph_dict, fp_dict = dict(), dict(), dict()

    isomers = gen_stereo(mol, nstereo)
    for i, mol in enumerate(isomers):
        mol = gen_conf(mol, nconf, seed)
        remove_confs(mol, energy, rms)

        phs = utils.load_multi_conf_mol(mol, smarts_features=pharm_def, bin_step=bin_step)
        mol_dict[i] = mol
        ph_dict[i] = [ph.get_feature_coords() for ph in phs]
        fp_dict[i] = [ph.get_fp() for ph in phs]
    return mol_name, mol_dict, ph_dict, fp_dict


def create_db(in_fname, out_fname, nconf, nstereo, energy, rms, ncpu, bin_step, pharm_def, seed, verbose):
    if verbose:
        sys.stderr.write('Database creation started\n')

    start_time = time.time()

    if out_fname.lower().endswith('.sdf.gz'):
        writer = gzip.open(out_fname, 'a')
        output_file_type = 'sdf.gz'
    elif out_fname.lower().endswith('.sdf'):
        writer = open(out_fname, 'at')
        output_file_type = 'sdf'
    elif out_fname.lower().endswith('.pkl'):
        writer = open(out_fname, 'wb')
        output_file_type = 'pkl'
    elif out_fname.lower().endswith('.dat'):
        db = DB(out_fname, flag='n')
        db.write_bin_step(bin_step)
        output_file_type = 'shelve'
    else:
        raise Exception("Wrong output file format. Can be only DAT, SDF, SDF.GZ or PKL.")

    nprocess = min(cpu_count(), max(ncpu, 1))
    p = Pool(nprocess)

    try:
        for i, (mol_name, mol_dict, ph_dict, fp_dict) in enumerate(
                p.imap_unordered(map_gen_data, prep_input(in_fname, nconf, nstereo, energy, rms, seed, bin_step,
                                                          pharm_def), chunksize=1), 1):
            if output_file_type == 'shelve':
                db.write_mol(mol_name, mol_dict)
                db.write_pharm(mol_name, ph_dict)
                db.write_fp(mol_name, fp_dict)

            elif output_file_type == 'pkl':
                for n, (mol, ph, fp) in enumerate(zip(mol_dict.values(), ph_dict.values(), fp_dict.values())):
                    pickle.dump((f'{mol_name}_{n}', mol, ph, fp), writer, -1)
            else:
                for n, (mol, ph, fp) in enumerate(zip(mol_dict.values(), ph_dict.values(), fp_dict.values())):
                    mol.SetProp("_Name", f'{mol_name}_{n}')
                    # mol.SetProp('pharm', '\n'.join(f'{label}\t{x}\t{y}\t{z}' for label, (x, y, z) in ph))
                    # mol.SetProp('fp', ','.join(map(str, sorted(fp))))
                    string = "$$$$\n".join(Chem.MolToMolBlock(mol, confId=c.GetId()) for c in mol.GetConformers())
                    if string:   # wrong molecules (no valid conformers) will result in empty string
                        string += "$$$$\n"
                        writer.write(string.encode("ascii") if output_file_type == 'sdf.gz' else string)
            if verbose and i % 10 == 0:
                current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                sys.stderr.write('\r{} molecules passed/conformers {}'.format(i, current_time))
                sys.stderr.flush()

    finally:
        p.close()

    if output_file_type != 'shelve':
        writer.close()
    # create new smi file if the input file has bad molecule structure(-s)
    else:
        if len(open(in_fname).readlines()) - 1 > len(db.get_mol_names()):
            sys.stderr.write("\nWarning: Some molecules were omitted from the original .smi file. All the original "
                             "molecules will be saved in a backup file\n")
            df_in = pd.read_csv(in_fname, sep='\t')
            cols = df_in.columns.tolist()
            df_in = df_in[df_in[cols[1]].isin(db.get_mol_names())]
            cols = df_in.columns.tolist()
            if len(cols) == 3:
                df_in = df_in.astype({cols[2]: int})
            file_name = os.path.basename(in_fname)
            pp_dirname = os.path.dirname(in_fname)
            file_name_new = f'#{file_name}.1#'
            for ll in os.listdir(pp_dirname):
                if os.path.isfile(os.path.join(pp_dirname, ll)) and os.path.splitext(file_name)[0] in ll and '#' in ll:
                    num = int(ll.split('.')[-1].split('#')[0])
                    pref = '.'.join(ll.split('.')[:-1])
                    file_name_new = pref + f'.{num + 1}#'
            os.rename(os.path.join(pp_dirname, file_name), os.path.join(pp_dirname, file_name_new))
            df_in.to_csv(os.path.join(os.path.dirname(in_fname), file_name), sep='\t', index=None)

    if verbose:
        sys.stderr.write("\n")


def entry_point():
    parser = argparse.ArgumentParser(description='Generate databased using RDKit.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--in_fname', metavar='FILENAME', required=True, type=str,
                        help='input file of 2D SDF or SMILES format (tab-separated).')
    parser.add_argument('-o', '--dbname', metavar='FILENAME.dat', required=True, type=str,
                        help='output database file name. Should have DAT extension. Database will consist of two files '
                             '.dat and .dir. If there is a database with the same name, then the tool will stop.')
    parser.add_argument('-b', '--bin_step', metavar='NUMERIC', type=int, default=1,
                        help='binning step for pharmacophores creation.')
    parser.add_argument('-s', '--nstereo', metavar='INTEGER', type=int, default=5,
                        help='maximum number of generated stereoisomers per compound (centers with specified '
                             'stereoconfogurations wil not be altered). ')
    parser.add_argument('-n', '--nconf', metavar='INTEGER', type=int, default=50,
                        help='number of generated conformers. ')
    parser.add_argument('-e', '--energy_cutoff', metavar='NUMERIC', type=float, default=None,
                        help='conformers with energy difference from the lowest one greater than the specified '
                             'threshold will be discarded.')
    parser.add_argument('-r', '--rms', metavar='NUMERIC', type=float, default=None,
                        help='only conformers with RMS higher then threshold will be kept. '
                             'Default: None (keep all conformers).')
    parser.add_argument('--seed', metavar='INTEGER', type=int, default=-1,
                        help='integer to init random number generator. Default: -1 (means no seed).')
    parser.add_argument('-p', '--pharm', metavar='FILENAME', type=str, default=None,
                        help='pharmacophore feature definition. If not specified default pmapper definitions '
                             'will be used.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', type=int, default=1,
                        help='number of cpu to use for calculation.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress to STDERR.')

    args = parser.parse_args()
    if (args.bin_step < 0) or (args.nstereo <= 0) or (args.nconf <= 0):
        sys.exit("--bin_step, --nstereo, --nconf can not be less 0.\n"
                 "--stereo and/or --nconf can not be set to 0, otherwise, the database will not be created correctly.")

    fdb = os.path.abspath(args.dbname)
    os.makedirs(os.path.dirname(os.path.abspath(fdb)), exist_ok=True)
    pdir, fname = os.path.dirname(fdb), os.path.basename(fdb)
    if fname in os.listdir(pdir):
        sys.exit("A database with this name already exists")

    create_db(in_fname=os.path.abspath(args.in_fname),
              out_fname=os.path.abspath(fdb),
              nconf=args.nconf,
              nstereo=args.nstereo,
              energy=args.energy_cutoff,
              rms=args.rms,
              bin_step=args.bin_step,
              pharm_def=args.pharm,
              ncpu=args.ncpu,
              seed=args.seed,
              verbose=args.verbose)


if __name__ == '__main__':
    entry_point()
