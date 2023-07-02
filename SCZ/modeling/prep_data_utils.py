import numpy as np
import nilearn
import pandas as pd
import os
import sys
import json
import concurrent.futures
from nilearn.connectome import sym_matrix_to_vec
from sklearn.feature_selection import SelectPercentile
import tqdm

diagnosis_mapping = {
    'CONTROL': 0,
    'SCHZ': 1,
    'Schizophrenia_Strict': 1,
    'No_Known_Disorder': 0,
    4: 1,
    0: 0
}

def threshMat(conn,lim): # if 5th percentile, then lim=95
	perc = np.array([np.percentile(x, lim) for x in conn])
	# Threshold each row of the matrix by setting values below X percentile to 0
	for i in range(conn.shape[0]):
		conn[i, conn[i,:] < perc[i]] = 0   
	return conn

def prepare_data_csv(data_paths, diag_mapping = None):
    data_csv = []
    for dataset in list(data_paths.keys()):
        participants = pd.read_csv(f'{data_paths[dataset]}/participants.tsv', sep='\t')[["participant_id", "diagnosis"]]
        participants["dataset"] = dataset
        participants["path"] = data_paths[dataset]
        data_csv.append(participants)
    data_csv = pd.concat(data_csv)
    if diag_mapping is not None:
        data_csv = data_csv[data_csv['diagnosis'].isin(list(diag_mapping.keys()))]
        data_csv["diagnosis"] = data_csv["diagnosis"].map(diag_mapping)
    data_csv["participant_id"] = data_csv["participant_id"].str.replace('sub-', '')
    return data_csv

def load_data(data_csv, data_type, comb_grads = None, n_grad = None, n_neighbours = None, aligned_grads = True, feat_selection = None, percentile = None):
    '''
    data_type: 'conn', 'disp', 'grad'
    '''

    data = []
    # add progress bar
    for subject in tqdm.tqdm(data_csv['participant_id']):
        root_path = data_csv[data_csv['participant_id'] == subject]['path'].values[0]
        subj_path = f'{root_path}/sub-{subject}/func'
        aligned = ''
        if data_type == 'grad':
            data_type = 'gradients'
            if aligned_grads:
                aligned = 'aligned'
        if data_type == 'disp' and comb_grads:
            data_type = f'disp-comb-{n_grad}grad-{n_neighbours}n'
        elif data_type == 'disp' and not comb_grads:
            data_type = f'disp-sing-{n_grad}grad-{n_neighbours}n'
        try:
            features = [np.load(f'{subj_path}/{i}') for i in os.listdir(subj_path) if data_type in i and aligned in i][0]
            if data_type == 'conn':
                if len(features.shape) == 3:
                    features = features[0]
                    np.fill_diagonal(features, 5)
                if feat_selection == 'value_percentile':
                    features = threshMat(features, percentile)
                    features = features[features != 0]         
                features = sym_matrix_to_vec(features, discard_diagonal=True)
            elif data_type == 'gradients':
                if comb_grads:
                    features = features[0,:, :n_grad].ravel()
                elif not comb_grads:
                    features = features[0,:, n_grad - 1]

            features = pd.DataFrame(features).T
            if feat_selection is not None:
                if percentile is None:
                        raise ValueError("Percentile must be specified for feature selection.")
                else:
                    if feat_selection == 'anova_percentile':
                        features.to_numpy()
                        print(features.shape)
                        features = SelectPercentile(percentile=percentile).fit_transform(features, data_csv['diagnosis'])
                        print(features.shape)
            
            features["participant_id"] = subject
            data.append(features)

        except FileNotFoundError as e:
            print(f"Data not found for subject {subject}: {e}.")
            data_csv = data_csv[data_csv['participant_id'] != subject]
    data = pd.concat(data,ignore_index=True, axis = 0)
    data = pd.merge(data, data_csv, on='participant_id')
    data = data.drop(columns=['participant_id', 'dataset', 'path'])
    return data, data_csv

def load_data_parallel(data_csv, data_type, comb_grads = None, n_grad = None, n_neighbours = None, aligned_grads = True):
    '''
    data_type: 'conn', 'disp', 'grad'
    '''
    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for subject in data_csv['participant_id']:
            futures.append(executor.submit(load_data_worker, subject, data_csv, data_type, comb_grads, n_grad, n_neighbours, aligned_grads))
        for future in concurrent.futures.as_completed(futures):
            data.append(future.result())
    data = pd.concat(data, ignore_index=True, axis=0)
    data = pd.merge(data, data_csv, on='participant_id')
    return data, data_csv

def load_data_worker(subject, data_csv, data_type, comb_grads, n_grad, n_neighbours, aligned_grads):
    root_path = data_csv[data_csv['participant_id'] == subject]['path'].values[0]
    subj_path = f'{root_path}/sub-{subject}/func'
    aligned = ''
    if data_type == 'grad':
        data_type = 'gradients'
        if aligned_grads:
            aligned = 'aligned'
    if data_type == 'disp' and comb_grads:
        data_type = f'disp-comb-{n_grad}grad-{n_neighbours}n'
    elif data_type == 'disp' and not comb_grads:
        data_type = f'disp-sing-{n_grad}grad-{n_neighbours}n'

    try:
        features = [np.load(f'{subj_path}/{i}') for i in os.listdir(subj_path) if data_type in i and aligned in i][0]
        if data_type == 'conn':
            if len(features.shape) == 3:
                features = features[0]
            features = sym_matrix_to_vec(features, discard_diagonal=True)
        elif data_type == 'gradients':
            if comb_grads:
                features = features[0,:, :n_grad].ravel()
            elif not comb_grads:
                features = features[0,:, n_grad - 1]
        features = pd.DataFrame(features).T
        features["participant_id"] = subject
        return features

    except FileNotFoundError as e:
        print(f"Data not found for subject {subject}: {e}.")
        data_csv = data_csv[data_csv['participant_id'] != subject]