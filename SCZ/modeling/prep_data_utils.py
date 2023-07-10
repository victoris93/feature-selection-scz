import numpy as np
import nilearn
import pandas as pd
import os
import sys
import json
import concurrent.futures
from nilearn.connectome import sym_matrix_to_vec
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
import tqdm

diagnosis_mapping = {
    'CONTROL': 0,
    'SCHZ': 1,
    'Schizophrenia_Strict': 1,
    'No_Known_Disorder': 0,
    4: 1,
    0: 0
}

def retain_top_columns(df, percentage):
    num_cols = int(df.shape[1] * percentage)
    top_cols = df.mean().nlargest(num_cols).index
    return df[top_cols]

def threshMat(conn,lim): # if 5th percentile, then lim=95
	perc = np.array([np.percentile(x, lim) for x in conn])
	# Threshold each row of the matrix by setting values below X percentile to 0
	for i in range(conn.shape[0]):
		conn[i, conn[i,:] < perc[i]] = 0   
	return conn

def prepare_data_csv(data_paths, diag_mapping = diagnosis_mapping):
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

def load_data(data_csv, data_type, comb_grads = False, n_grad = None, n_neighbours = None, aligned_grads = True, feat_selection = None, percentile = None, nbs_thresh = None, nbs_dir = None):
    '''
    data_type: 'conn', 'disp', 'grad', 'nbs'
    '''

    data = []
    # add progress bar
    if feat_selection is not None:
        if percentile is None and nbs_thresh is None:
            raise ValueError("Either percentile or nbs_thresh must be specified for feature selection.")
                
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
                while len(features.shape) > 2:
                    features = features[0]
                np.fill_diagonal(features, 0)
                features = sym_matrix_to_vec(features, discard_diagonal=True)
                if feat_selection == 'nbs':
                    nbs_thresh = float(nbs_thresh)
                    if nbs_dir is None:
                        nbs_dir = os.getcwd()
                    adj = np.load(f'{nbs_dir}/nbs_{nbs_thresh}thresh.npy', allow_pickle=True)[1]
                    adj = sym_matrix_to_vec(adj, discard_diagonal=True)
                    features = features[adj != 0]

            elif data_type == 'gradients':
                if comb_grads:
                    features = features[0,:, :n_grad].ravel()
                elif not comb_grads:
                    features = features[0,:, n_grad - 1]
            elif data_type == 'nbs':
                adj = features
            data.append(features)

        except FileNotFoundError as e:
            print(f"Data not found for subject {subject}: {e}.")
            data_csv = data_csv[data_csv['participant_id'] != subject]

    data = np.row_stack(data)
    if feat_selection == 'anova_percentile':
            print(data.shape)
            data = SelectPercentile(percentile=percentile).fit_transform(data, data_csv['diagnosis'].values)
            print(data.shape)
    data = pd.DataFrame(data)
    
    if feat_selection == 'value_percentile':
        data = retain_top_columns(data, percentile/100)
    data["diagnosis"] = data_csv['diagnosis'].values
    #data = data.dropna()
    return data, data_csv

def load_data_pca(participants, path_to_args):
    all_data = []
    args = np.loadtxt(path_to_args, dtype=str)

    for row in args:
        data_type = row[0]
        comb_grads = bool(row[1])

        n_grad = row[2]
        n_grad = None if n_grad == 'None' else int(n_grad)

        n_neighbours =row[3]
        n_neighbours = None if n_neighbours == 'None' else int(n_neighbours)

        aligned_grads = bool(row[4])

        feat_selection = row[5]
        feat_selection = None if feat_selection == 'None' else feat_selection

        percentile = row[6]
        percentile = None if percentile == 'None' else float(percentile)

        nbs_thresh = row[7]
        nbs_thresh = None if nbs_thresh == 'None' else float(nbs_thresh)

        nbs_dir = row[8]
        nbs_dir = None if nbs_dir == 'None' else nbs_dir

        print("Loading features for data type: ", data_type)
        data, participants = load_data(participants, data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile)
        data = data.drop(columns=['diagnosis'])
        all_data.append(data)

    all_data = pd.concat(all_data, axis=1, ignore_index=True)
    all_data = all_data.to_numpy()
    pca = PCA()
    print("Running PCA on all features...")
    all_data = pca.fit_transform(all_data)
    all_data = pd.DataFrame(all_data)
    all_data["diagnosis"] = participants['diagnosis'].values
    print("Data loaded.")
    return all_data, participants