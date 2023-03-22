#!/usr/bin/env python3
# author          : Alina Kutlushina
# date            : 10.01.2019
# license         : BSD-3
# ==============================================================================

import os
import sys
import argparse
from multiprocessing import Pool
from psearch.screen_db import screen_db
from psearch.scripts.external_statistics import calc_stat
from psearch.scripts.gen_pharm_models import gen_pharm_models
from psearch.scripts.select_training_set import trainingset_formation


def create_parser():
    parser = argparse.ArgumentParser(description='Ligand-based pharmacophore model building',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--project_dir', metavar='DIRNAME', type=str, default=None,
                        help='A path to a project dir. Directory where all intermediate and output files will be saved.')
    parser.add_argument('-i', '--molecules', metavar='FILENAME.smi', type=str, required=True,
                        help='The script takes as input a tab-separated SMILES file containing smiles, compound id and '
                             'activity columns. The third column should contain a word 1 or 0. 1 is for actives, '
                             '0 is for inactives.')
    parser.add_argument('-d', '--database', metavar='FILENAME.dat', type=str, required=True,
                        help='Path to the database with precomputed conformers and pharmacophores for the same input file.')
    parser.add_argument('-ts', '--trainset', metavar='DIRNAME', type=str, default=None,
                        help='A path to the folder where will be saved a training set.'
                             'If omitted, the path will be generated automatically relative to project directory.')
    parser.add_argument('-q', '--query', metavar='DIRNAME', type=str, default=None,
                        help='A path to a folder where will be saved the created pharmacophore models.'
                             'If omitted, the path will be generated automatically relative to project directory.')
    parser.add_argument('-s', '--screening', metavar='DIRNAME', type=str, default=None,
                        help='In the screen folder will be stored the results of virtual screening on the input database'
                             ' using the created models. This is needed for calculation external statistics. '
                             'It is a step of the created pharmacophore models validation. '
                             'If omitted, the path will be generated automatically relative to project directory.')
    parser.add_argument('-r', '--external_statistics', metavar='FILENAME', default=None,
                        help='An output text file where will be saved validation statistics. '
                             'If omitted, the path will be generated automatically relative to project directory.')
    parser.add_argument('-m', '--mode_train_set', nargs='+', type=int, default=[1, 2],
                        help='Take numbers 1 or 2 or both to designate the strategy to create training sets. '
                             '1 - a single training set will be created from centroids of individual clusters, '
                             '2 - multiple training sets will be created, one per cluster.')
    parser.add_argument('--fcfp4', action='store_true', default=False,
                        help='If set FCFP4 fingerprints will be used for compound clustering, '
                             'otherwise pharmacophore fingerprints will be used.')
    parser.add_argument('-t', '--threshold', metavar='NUMERIC', type=float, default=0.4,
                        help='threshold for сlustering data by Butina algorithm.')
    parser.add_argument('-tol', '--tolerance', metavar='NUMERIC', type=float, default=0,
                        help='tolerance used for calculation of a stereoconfiguration sign.')
    parser.add_argument('-b', '--bin_step', metavar='NUMERIC', type=float, default=1,
                        help='binning step.')
    parser.add_argument('-l', '--lower', metavar='INTEGER', type=int, default=3,
                        help='starting from this number of features, pharmacophore models will be created')
    parser.add_argument('-f', '--save_model_complexity', metavar='INTEGER', type=int, default=None,
                        help='All pharmacophore models will be saved starting from this number of features.'
                             'If omitted will be saved only the most complex pharmacophore models')
    parser.add_argument('-u', '--upper', metavar='INTEGER', type=int, default=None,
                        help='limit the upper number of features in generated pharmacophores. '
                             'If omitted pharmacophores of maximum complexity will be generated.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', type=int, default=1,
                        help='number of cpus to use for calculation.')
    return parser


def creating_pharmacophore_mp(items):
    return creating_pharmacophore(*items)


def get_items(in_db, list_ts, path_pma, upper, lower, save_model_complexity, bin_step, tolerance, save_stat):
    for train_set in list_ts:
        yield in_db, train_set, path_pma, upper, lower, save_model_complexity, bin_step, tolerance, save_stat


def creating_pharmacophore(in_db, train_set, path_pma, upper, lower, save_model_complexity, bin_step, tolerance, save_stat):
    gen_pharm_models(in_db=in_db,
                     trainset=train_set,
                     out_pma=path_pma,
                     upper=upper,
                     bin_step=bin_step,
                     tolerance=tolerance,
                     current_nfeatures=lower,
                     nfeatures=save_model_complexity,
                     save_statistics=save_stat)


def main(in_mols, in_db, path_ts, path_pma, path_screen, path_external_stat, path_clus_stat,
         mode_train_set, fcfp4, threshold, tolerance, lower, save_model_complexity, upper, bin_step, ncpu, save_stat):

    # formation of a training set
    list_ts = trainingset_formation(input_mols=in_mols,
                                    path_ts=path_ts,
                                    mode_train_set=mode_train_set,
                                    fcfp4=fcfp4,
                                    clust_stat=open(path_clus_stat, 'w'),
                                    threshold=threshold)

    if type(list_ts) == str:
        sys.exit(list_ts)

    p = Pool(ncpu)
    for _ in p.imap(creating_pharmacophore_mp, get_items(in_db=in_db, list_ts=list_ts, path_pma=path_pma,
                                                         upper=upper, lower=lower,
                                                         save_model_complexity=save_model_complexity,
                                                         bin_step=bin_step, tolerance=tolerance, save_stat=save_stat)):
        continue
    p.close()

    # validation of the created pharmacophore queries
    screen_db(db_fname=in_db,
              queries=[os.path.join(path_pma, mm) for mm in os.listdir(path_pma)],
              output=path_screen, output_sdf=None,
              match_first_conf=True, min_features=None,
              ncpu=ncpu, verbose=True)

    calc_stat(path_mols=in_mols,
              path_ts=path_ts,
              pp_models=path_pma,
              path_screen=path_screen,
              out_external=path_external_stat)


def entry_point():
    parser = create_parser()
    args = parser.parse_args()
    project_dir = os.path.abspath(args.project_dir) if args.project_dir else os.path.dirname(os.path.abspath(args.molecules))
    os.makedirs(project_dir, exist_ok=True)
    pp_model = os.path.abspath(args.query) if args.query else os.path.join(project_dir, 'models')
    os.makedirs(pp_model, exist_ok=True)
    main(in_mols=os.path.abspath(args.molecules),
         in_db=os.path.abspath(args.database),
         path_ts=os.path.abspath(args.trainset) if args.trainset else os.path.join(project_dir, 'trainset'),
         path_pma=pp_model,
         path_screen=os.path.abspath(args.screening) if args.screening else os.path.join(project_dir, 'raw_screen'),
         path_external_stat=os.path.abspath(args.external_statistics) if args.external_statistics else
                            os.path.join(project_dir, 'external_statistics.txt'),
         path_clus_stat=os.path.join(pp_model, 'cluster_stat_trh{}.txt'.format(args.threshold)),
         mode_train_set=args.mode_train_set,
         fcfp4=args.fcfp4,
         threshold=float(args.threshold),
         tolerance=float(args.tolerance),
         bin_step=int(args.bin_step),
         lower=int(args.lower),
         save_model_complexity=int(args.save_model_complexity) if args.save_model_complexity else None,
         upper=int(args.upper) if args.upper is not None else None,
         ncpu=int(args.ncpu),
         save_stat=False)


if __name__ == '__main__':
    entry_point()
