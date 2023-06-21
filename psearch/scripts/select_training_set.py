#!/usr/bin/env python3

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from pmapper.customize import load_factory


def generate_fingerprints(smiles, fcfp4):
    if fcfp4:
        return (AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2, useFeatures=True))
    featfactory = load_factory()
    sigfactory = SigFactory(featfactory, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
    sigfactory.SetBins([(0, 2), (2, 5), (5, 8)])
    sigfactory.Init()
    return (Generate.Gen2DFingerprint(Chem.MolFromSmiles(smiles), sigfactory))
    

def gen_cluster_subset_butina(fps, cutoff):
    dists = []
    for i in range(len(fps)-1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        dists.extend([1 - x for x in sims])
    clusters = [list(c) for c in Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)]
    return clusters  # returns tuple of tuples with sequential numbers of compounds in each cluster


def save_cluster_stat(cs_index, df, index_acts, clust_stat):
    clust_stat.write("""#The saved file contains statistical information about the cluster 
#and the index of the molecule according to its location in the input file \n""")
    for i, cluster in enumerate(cs_index):
        i_act = len(set(cluster).intersection(index_acts))
        clust_stat.write(f'\ncluster {i}, cluster length {len(cluster)}, share of active {round(i_act/len(cluster), 3)}\n')
        clust_stat.write(','.join(map(str, cluster)) + '\n')
        
def get_cluster_stat(clusters_set, df):
    clust_stat_data = []
    index_actives = set(df[df.activity == 1].index)
    for i, cluster in enumerate(clusters_set):
        actives_size = len(set(cluster) & index_actives)
        clust_stat_data.append(dict(cluster_id=i,
                                    cluster_length=len(cluster),
                                    actives_share=round(actives_size/len(cluster), 3),
                                    elements=cluster))
    return clust_stat_data

            
def select_data(df, clusters_all, num_elements):
    index_actives = set(df[df.activity == 1].index)
    index_inactives = set(df[df.activity == 0].index)
    for i, cluster in enumerate(clusters_all):
        actives_ids = list(set(cluster) & index_actives)
        inactives_ids = list(set(cluster) & index_inactives)
        if len(actives_ids) >= num_elements:
            actives = df[df.index.isin(actives_ids)].head(num_elements).values.tolist()
            inactives = df[df.index.isin(inactives_ids)].head(num_elements).values.tolist()
            yield i, actives, inactives 


def get_centroids(cs, df, num):
    return [df.iloc[x[0]].tolist() for x in cs if len(x) > num]


def generate_training_set(activity_df, training_set_mode, fcfp4, threshold, clust_size=5, min_num_acts=5):

    if (1 not in training_set_mode) and (2 not in training_set_mode):
        raise RuntimeError('Wrong value of parameter mode_train_set. That should be 1 and/or 2.')

    activity_df["fingerprints"] = activity_df.smiles.apply(generate_fingerprints, args=[fcfp4])
    
    columns = ["smiles", "mol_name", "activity"]
    
    # getting clusters for all molecules and only (in)active
    clusters_all = gen_cluster_subset_butina(activity_df.fingerprints.to_list(), cutoff=threshold)
    clusters_active = gen_cluster_subset_butina(activity_df[activity_df.activity == 1].fingerprints.to_list(), cutoff=threshold)
    clusters_inactive = gen_cluster_subset_butina(activity_df[activity_df.activity == 0].fingerprints.to_list(), cutoff=threshold)
    
    # rescale clusters members to match DF indecies
    border_index = activity_df[activity_df.activity==0].index.min()
    clusters_inactive_rescaled = [[x+border_index for x in cluster] for cluster in clusters_inactive]
    
    # getting centroids af the clusters
    centroids_active = get_centroids(clusters_active, activity_df[columns], clust_size)
    centroids_inactive = get_centroids(clusters_inactive_rescaled, activity_df[columns], clust_size)
        
    training_set = []
    if 2 in training_set_mode:
        for i, actives, inactives in select_data(activity_df[columns], clusters_all, min_num_acts):
            training_set.append((f"t{i}", actives+inactives+centroids_inactive))
            
    clust_stat_data = get_cluster_stat(clusters_all, activity_df)
    
    if 1 in training_set_mode:
        if len(centroids_active) < min_num_acts:
            return training_set, clust_stat_data        
        
        training_set.append(("centroids", centroids_active+centroids_inactive))

    return training_set, clust_stat_data
