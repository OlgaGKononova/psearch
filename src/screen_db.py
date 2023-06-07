#!/usr/bin/env python3
# author          : Pavel Polishchuk
# date            : 23.08.2019
# license         : BSD-3
# ==============================================================================

import os
import sys
import time
from collections import namedtuple
from pmapper.pharmacophore import Pharmacophore
from multiprocessing import Pool
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem


path_query = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pharmacophores', 'chembl_models')
Model = namedtuple('Model', ['name', 'fp', 'pharmacophore'])
Conformer = namedtuple('Conformer', ['stereo_id', 'conf_id', 'fp', 'pharmacophore'])

def load_confs(mol_name, db):
    bin_step = db.get_bin_step()
    fp_dict = db.get_fp(mol_name)
    ph_dict = db.get_pharm(mol_name)
    res = []
    for stereo_id in fp_dict:
        try:
            for conf_id, (fp, coord) in enumerate(zip(fp_dict[stereo_id], ph_dict[stereo_id])):
                p = Pharmacophore(bin_step=bin_step)
                p.load_from_feature_coords(coord)
                res.append(Conformer(stereo_id, conf_id, fp, p))
        except:
            print(mol_name)
    return res


def read_models(queries, bin_step, min_features):
    """_summary_

    Args:
        queries (_type_): List of {'bin_size': 1,
                                    'coords': [('a', (-2.68, 1.25, 5.07)),
                                                ('a', (3.58, -4.65, -1.1)),
                                                ('H', (-2.68, 1.25, 5.07)),
                                                ('H', (3.58, -4.65, -1.1))],
                                    'name': 'DB_NAME.t0_f5_p0.xyz'}
        output (_type_): _description_
        bin_step (_type_): _description_
        min_features (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    res = []
    for q_data in queries:
        p = Pharmacophore()
        p.load_from_feature_coords(tuple(q_data["coords"]))
        # skip models with less number of features with distinct coordinates that given
        if min_features is not None and len(set(xyz for label, xyz in p.get_feature_coords())) < min_features:
            continue
        p.update(bin_step=bin_step)
        fp = p.get_fp()
        res.append(Model(q_data["name"], fp, p))

    return res


def screen(mol_name, db, models, match_first_conf, get_transform_matrix=True, get_rms=True):

    def compare_fp(query_fp, fp):
        return (query_fp & fp) == query_fp

    confs = load_confs(mol_name, db)

    output = []
    for model in models:
        for conf in confs:
            if compare_fp(model.fp, conf.fp):
                res = conf.pharmacophore.fit_model(model.pharmacophore,
                                                   get_transform_matrix=get_transform_matrix, get_rms=get_rms)
                if res:
                    if get_transform_matrix:
                        output.append((model.name, mol_name, conf.stereo_id, conf.conf_id, res[1], res[2]))
                    else:
                        output.append((model.name, mol_name, conf.stereo_id, conf.conf_id))
                    if match_first_conf:
                        break
    return output


def save_results(results, output_sdf, db):
    for items in results:
        mol_name, stereo_id, conf_id, out_fname = items[:4]
        if not os.path.exists(os.path.dirname(out_fname)):
            os.makedirs(os.path.dirname(out_fname))
        with open(out_fname, 'at') as f:
            f.write('\t'.join((mol_name, str(stereo_id), str(conf_id))) + '\n')
    if output_sdf:
        for mol_name, stereo_id, conf_id, out_fname, matrix, rms in results:
            # print('!'*8, type(rms), rms)
            m = db.get_mol(mol_name)[stereo_id]
            AllChem.TransformMol(m, matrix, conf_id)
            m.SetProp('_Name', f'{mol_name}-{stereo_id}-{conf_id}')
            m.SetProp("RMSD", str(round(rms, 4)))
            with open(os.path.splitext(out_fname)[0] + '.sdf', 'a') as f:
                w = Chem.SDWriter(f)
                w.write(m)
                w.close()


def screen_db(db, queries, match_first_conf, min_features, ncpu, verbose):

    start_time = time.time()

    bin_step = db.get_bin_step()
    models = read_models(queries, bin_step, min_features)   # return list of Model namedtuples

    comp_names = db.get_mol_names()
    
    screen_output = {}

    if ncpu == 1:
        for i, comp_name in enumerate(comp_names, 1):
            res = screen(mol_name=comp_name, db=db, models=models, match_first_conf=match_first_conf)
            if res:
                for model_name, mol_name, a, b, m, r in res:
                    if model_name not in screen_output:
                        screen_output[model_name] = []
                    screen_output[model_name].append((mol_name, a, b))
            if verbose and i % 10 == 0:
                current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                sys.stderr.write('\r{} molecules passed/conformers {}'.format(i, current_time))
                sys.stderr.flush()
    else:
        p = Pool(ncpu)
        for i, res in enumerate(p.imap_unordered(partial(screen, db=db, models=models, match_first_conf=match_first_conf), comp_names, chunksize=10), 1):
            
            if res:
                for model_name, mol_name, a, b, m, r in res:
                    if model_name not in screen_output:
                        screen_output[model_name] = []
                    screen_output[model_name].append((mol_name, a, b))
            if verbose and i % 10 == 0:
                current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                sys.stderr.write('\r{} molecules screened {}'.format(i, current_time))
                sys.stderr.flush()
        p.close()
    return screen_output

