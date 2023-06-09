#!/usr/bin/env python3
# author          : Alina Kutlushina
# date            : 10.01.2019
# license         : BSD-3
# ==============================================================================

from multiprocessing import Pool
from src.scripts.gen_pharm_models import gen_pharm_models


def creating_pharmacophore_mp(items):
    return creating_pharmacophore(*items)


def get_items(db, train_set, upper, lower, save_model_complexity, bin_step, tolerance):
    for train_batch in train_set:
        yield db, train_batch, upper, lower, save_model_complexity, bin_step, tolerance


def creating_pharmacophore(db, training_set, upper, lower, save_model_complexity, bin_step, tolerance):
    return gen_pharm_models(db=db,
                     training_set=training_set,
                     upper=upper,
                     bin_step=bin_step,
                     tolerance=tolerance,
                     current_nfeatures=lower,
                     nfeatures=save_model_complexity)


def psearch_generator(db, training_set, tolerance, lower, save_model_complexity, upper, bin_step, ncpu):
        
    p = Pool(ncpu)
    output = []
    for data in p.imap(creating_pharmacophore_mp, get_items(db=db, train_set=training_set, 
                                                         upper=upper, lower=lower,
                                                         save_model_complexity=save_model_complexity,
                                                         bin_step=bin_step, tolerance=tolerance)):
        output.extend(data)
    p.close()
    
    return output

