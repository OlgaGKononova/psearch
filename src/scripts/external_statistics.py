#!/usr/bin/env python3
# author          : Alina Kutlushina
# date            : 01.05.2018
# license         : BSD-3
# ==============================================================================

import time
import pandas as pd
from rdkit import Chem
from pmapper.pharmacophore import Pharmacophore as P


def max_edge(model):
    coords = model["coords"]
    edge = 0 
    for i, c1 in enumerate(coords):
        for j, c2 in enumerate(coords[i + 1:]):
            e = ((c1[1][0] - c2[1][0]) ** 2 + (c1[1][1] - c2[1][1]) ** 2 + (c1[1][2] - c2[1][2]) ** 2) ** (1 / 2)
            if e > edge:
                edge = e
    return edge


def get_external_stat(df_mols, trainset, pharm_model, pp_screen, model_id):
    target_id = "input_smiles" #os.path.splitext(os.path.basename(path_mols))[0]
    medge = max_edge(pharm_model)
    num_uniq_features = set()
    labels = ''
    
    for label, coords in pharm_model["coords"]:
        labels += label
        num_uniq_features.add(tuple(map(float, coords)))
    num_uniq_features = len(num_uniq_features)

    ts_act_mol = []
    ts_inact_mol = []
    for smiles, name, activity in trainset:
        if activity == 1:
            ts_act_mol.append(name)
        else:
            ts_inact_mol.append(name)

    if not Chem.MolFromSmiles(df_mols.at[0, 'smiles']):
        print("dropped:", df_mols.at[0, 'smiles'])
        df_mols.drop(index=0, inplace=True)
    df_act = df_mols[(df_mols['activity'] == 1) & (~df_mols['mol_name'].isin(ts_act_mol))]
    df_inact = df_mols[(df_mols['activity'] == 0) & (~df_mols['mol_name'].isin(ts_inact_mol))]
    
    res_screen = [name for name, _, _ in pp_screen]
    act_screen = set(res_screen) & set(df_act['mol_name'])
    inact_screen = set(res_screen) & set(df_inact['mol_name'])

    p = df_act.shape[0]
    n = df_inact.shape[0]
    tp = len(act_screen)
    fp = len(inact_screen)
    tn = n - fp

    try:
        recall = tp / p
    except ZeroDivisionError:
        recall = None
    try:
        tnr = tn / n
        fpr = fp / n
    except ZeroDivisionError:
        tnr, fpr = None, None
    if recall and tnr:
        ba = (recall + tnr) / 2
    else:
        ba = None

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = None

    if precision:
        ef = precision / (p / (p + n))
        f1 = (2 * precision * recall) / (precision + recall)
        f2 = (5 * precision * recall) / (4 * precision + recall)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    else:
        ef, f1, f2, f05 = None, None, None, None
    return target_id, model_id, tp, fp, p, n, precision, recall, fpr, f1, f2, f05, ba, ef, num_uniq_features, medge, labels


def calc_stat(activity_df, training_set, pharm_data, screen_out):
    start_time = time.time()
    df_result = pd.DataFrame(columns=['target_id', 'model_id', 'TP', 'FP', 'P', 'N', 'precision', 'recall', 'FPR',
                                      'F1', 'F2', 'F05', 'BA', 'EF', 'uniq_features', 'max_dist', 'features'])
    for enum, fmodel in enumerate(pharm_data):
        target_id, model_id = fmodel["name"].split(".")
        set_id = model_id.split("_")[0]
        results = get_external_stat(df_mols=activity_df,
                                    trainset=training_set[set_id],
                                    pharm_model=fmodel, 
                                    pp_screen=screen_out[fmodel["name"]],
                                    model_id=model_id)
        if results:
            df_result.loc[enum] = results
        else:
            continue

    df_result = df_result.sort_values(by=['recall', 'F05', 'F2'], ascending=False)
    df_result = round(df_result, 3)
    return df_result
