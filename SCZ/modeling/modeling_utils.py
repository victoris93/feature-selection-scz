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
from pycaret.classification import *
import tqdm
import multiprocessing
from scipy.stats import percentileofscore


diagnosis_mapping = {
    'CONTROL': 0,
    'SCHZ': 1,
    'Schizophrenia_Strict': 1,
    'No_Known_Disorder': 0,
    4: 1,
    0: 0
}

sex_mapping = {
    1: 0,
    2: 1,
    'M': 0,
    'F': 1,
    'male': 0,
    'female': 1
}

def str2bool(value):
  return value.lower() in ("yes", "true", "t", "1")

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
        participants = pd.read_csv(f'{data_paths[dataset]}/participants.tsv', sep='\t')[["participant_id", "diagnosis", 'age', 'sex']]
        participants["dataset"] = dataset
        participants["path"] = data_paths[dataset]
        if "COBRE" in participants["dataset"].unique()[0]:
            participants["dataset"] = "COBRE"
        data_csv.append(participants)
    data_csv = pd.concat(data_csv, ignore_index=True, verify_integrity=True)
    if (0, 1) not in data_csv["sex"].unique():
            data_csv["sex"] = data_csv["sex"].replace(1, 0)
            data_csv["sex"] = data_csv["sex"].replace('M', 0)
            data_csv["sex"] = data_csv["sex"].replace('F', 1)
            data_csv["sex"] = data_csv["sex"].replace('male', 0)
            data_csv["sex"] = data_csv["sex"].replace('female', 1)
            data_csv["sex"] = data_csv["sex"].replace(2, 1)
    if diag_mapping is not None:
        data_csv = data_csv[data_csv['diagnosis'].isin(list(diag_mapping.keys()))]
        data_csv["diagnosis"] = data_csv["diagnosis"].map(diag_mapping)
    data_csv["participant_id"] = data_csv["participant_id"].str.replace('sub-', '')
    for row in data_csv.iloc:
        subject = row["participant_id"]
        subj_path = f'{row["path"]}/sub-{subject}/func'
        if not os.path.exists(subj_path):
            data_csv = data_csv[data_csv['participant_id'] != subject]
            
    return data_csv

def load_data(data_csv, data_type, comb_grads = False, n_grad = None, n_neighbours = None, aligned_grads = True, feat_selection = None, percentile = None, nbs_thresh = None, nbs_dir = None, format = 'numpy'):
    '''
    data_type: 'conn', 'disp', 'grad', 'nbs'
    '''

    data = []
    # add progress bar
    if feat_selection is not None:
        if percentile is None and nbs_thresh is None:
            raise ValueError("Either percentile or nbs_thresh must be specified for feature selection.")
                
    for subject in data_csv['participant_id']:
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
                elif not comb_grads or comb_grads is None:
                    features = features[0,:, n_grad - 1]
            elif data_type == 'nbs':
                adj = features
            data.append(features)

        except FileNotFoundError as e:
            print(f"Data not found for subject {subject}: {e}.")
            data_csv = data_csv[data_csv['participant_id'] != subject]

    data = np.row_stack(data)
    if feat_selection == 'anova_percentile':
            data = SelectPercentile(percentile=percentile).fit_transform(data, data_csv['diagnosis'].values)
    if format == 'pandas':
        data = pd.DataFrame(data)
        
        if feat_selection == 'value_percentile':
            data = retain_top_columns(data, percentile/100)
        data["diagnosis"] = data_csv['diagnosis'].values
    #data = data.dropna()
    return data, data_csv

def parse_args(args):
    data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir = args

    comb_grads = None if comb_grads == "None" else str2bool(comb_grads)
    n_grad = None if n_grad == 'None' else int(n_grad)
    n_neighbours = None if n_neighbours == 'None' else int(n_neighbours)
    aligned_grads = None if aligned_grads == 'None' else str2bool(aligned_grads)
    feat_selection = None if feat_selection == 'None' else feat_selection
    percentile = None if percentile == 'None' else float(percentile)
    nbs_thresh = None if nbs_thresh == 'None' else float(nbs_thresh)
    nbs_dir = None if nbs_dir == 'None' else nbs_dir

    return data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir

def parse_feat_id(args):
    args = tuple(args[:-1])
    data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh = args
    if 'None' not in comb_grads:
        comb_grads = str2bool(comb_grads)
        if comb_grads:
            comb_grads = 'comb'
        else:
            comb_grads = 'sing'
    if 'None' not in aligned_grads:
        aligned_grads = str2bool(aligned_grads)
        if aligned_grads:
            aligned_grads = 'aligned'
        else:
            aligned_grads = 'unaligned'
    id_args = []
    for arg in data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh:
        if 'None' not in arg:
            id_args.append(arg)
    feature_id = '_'.join(id_args)
    return feature_id

def load_features_pca(data_csv, path_to_args):
    all_data = []
    args = np.loadtxt(path_to_args, dtype=str)
    features_ids = []
    for row in args:
        data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir = parse_args(row)
        feature_id = parse_feat_id(row)
        print("Loading features for data type: ", data_type)
        data, data_csv = load_data(data_csv, data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir, "pandas")
        features_ids.extend([feature_id] * len(data.columns))
        data = data.drop(columns=['diagnosis'])
        all_data.append(data)

    all_data = pd.concat(all_data, axis=1, ignore_index=True)
    all_data = all_data.to_numpy()
    pca = PCA()
    print("Running PCA on all features...")
    all_data = pca.fit_transform(all_data)
    all_data = pd.DataFrame(all_data)
    all_data["diagnosis"] = data_csv['diagnosis'].values
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    print("Feature PCs loaded.")
    return all_data, eigenvectors, eigenvalues, data_csv, features_ids

def load_all_features(data_csv, path_to_args):
    all_data = []
    args = np.loadtxt(path_to_args, dtype=str)
    features_ids = []
    for row in args:
        data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir = parse_args(row)
        feature_id = parse_feat_id(row)
        print("Loading features for data type: ", data_type)
        data, data_csv = load_data(data_csv, data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir)
        features_ids.extend([feature_id] * len(data.columns))
        data = data.drop(columns=['diagnosis'])
        all_data.append(data)
    all_data = pd.concat(all_data, axis=1, ignore_index=True)
    columns = all_data.columns
    new_columns = []
    for i, column in enumerate(columns):
        new_columns.append(f"{column}_{features_ids[i]}")
    all_data.columns = new_columns
    print("Features loaded.")
    return all_data, data_csv

def get_n_best_features(feature_importance_matrix, n, features, feature_labels):
    feature_importance_matrix = feature_importance_matrix
    max_values = np.max(feature_importance_matrix, axis=1)
    top_indices = np.argsort(-max_values)[:n]
    best_features = features.iloc[:, top_indices]
    feature_labels = feature_labels[top_indices]
    best_features.columns = feature_labels
    return best_features

def get_n_random_features(n, features):
    feat_indices = np.random.choice(np.arange(features.shape[1]), n, replace=False)
    random_features = features.iloc[:, feat_indices]
    return random_features

def fit_on_best_features(best_features):
    experiment = setup(best_features, target = 'diagnosis', session_id = 1, verbose=False)
    best_fit = compare_models(verbose=False)
    best_fit = pull()
    del experiment
    return best_fit

def fit_on_random_features(args): # parallelize
    n_features, features, data_csv, seed = args # best_acc and best_auc are mean across all tested models
    random_features = get_n_random_features(n_features, features)
    random_features["diagnosis"] = data_csv["diagnosis"].values
    experiment = setup(random_features, target = 'diagnosis', session_id = seed, verbose=False)
    print(f"Test {seed + 1}...")
    lr = create_model('lr')
    performance = pull()
    del lr
    del experiment
    dummy_acc = performance[performance["Model"] == "Dummy Classifier"]["Accuracy"].values[0]
    dummy_auc = performance[performance["Model"] == "Dummy Classifier"]["AUC"].values[0]

    performance = performance[performance["Accuracy"] > dummy_acc]
    mean_acc = performance["Accuracy"].mean()
    performance = performance[performance["AUC"] > dummy_auc]
    mean_auc = performance["AUC"].mean()

    return mean_acc, mean_auc

def random_feature_test(n_features, features, best_acc, best_auc, data_csv, workers, n_tests = 1000): # parallelize
    null_acc = []
    null_auc = []

    print("Fitting models on randomly picked features...")
    if workers == -1:
        workers = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(workers)
    random_test_args = [(n_features, features, data_csv, test) for test in range(n_tests)]
    null_dists = pool.map(fit_on_random_features, random_test_args)
    
    pool.close()
    pool.join()
    
    for values in null_dists:
        null_acc.append(values[0])
        null_auc.append(values[1])

    null_acc_p = 1 - percentileofscore(null_acc, best_acc) / 100
    null_auc_p = 1 - percentileofscore(null_auc, best_auc) / 100

    return null_acc_p, null_auc_p

def aggregate_data(data_csv, path_to_args):
    args = np.loadtxt(path_to_args, dtype=str)
    for row in tqdm.tqdm(args):
        data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir = parse_args(row)
        feature_id = parse_feat_id(row)
        print("Loading features for data type: ", data_type)
        data, data_csv = load_data(data_csv, data_type, comb_grads, n_grad, n_neighbours, aligned_grads, feat_selection, percentile, nbs_thresh, nbs_dir)
        features_ids =[feature_id] * data.shape[1]
        features_ids = ["_".join([i, str(j)]) for i, j in zip(features_ids, np.arange(data.shape[1]))]
        if not os.path.exists("all_features.npy"):
            np.save("all_features", data)
        else:
            all_features = np.load("all_features.npy")
            all_features = np.concatenate([all_features, data], axis=1)
            np.save("all_features", all_features)
            print(all_features.shape)
        if not os.path.exists("feature_labels.npy"):
            np.save("feature_labels", np.array(features_ids))
        else:
            feature_labels = np.load("feature_labels.npy")
            feature_labels = np.concatenate([feature_labels, np.array(features_ids)])
            np.save("feature_labels", feature_labels)
            print(feature_labels.shape)
    print(f"Features aggregated in all_features.npy ({all_features.shape}), labels in feature_labels.npy ({feature_labels.shape})")
