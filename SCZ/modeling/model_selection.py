import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
import sys
import csv
from prep_data_utils import *
from pycaret.classification import *
from nilearn.connectome import sym_matrix_to_vec

data_type = sys.argv[1]
feat_selection = sys.argv[2]
feat_selection = None if feat_selection == 'None' else feat_selection
percentile = sys.argv[3]
percentile = None if percentile == 'None' else float(percentile)
n_neighbours = sys.argv[4]
n_neighbours = None if n_neighbours == 'None' else int(n_neighbours)
n_grad = sys.argv[5]
n_grad = None if n_grad == 'None' else int(n_grad)
comb_grads = sys.argv[6]
comb_grads = None if comb_grads == 'None' else bool(comb_grads)

diagnosis_mapping = {
    'CONTROL': 0,
    'SCHZ': 1,
    'Schizophrenia_Strict': 1,
    'No_Known_Disorder': 0,
    4: 1,
    0: 0
}

# load data
data_paths = json.load(open('data_paths.json', 'r'))
data_csv = prepare_data_csv(data_paths, diag_mapping = diagnosis_mapping)
data, _ = load_data(data_csv, data_type, comb_grads = comb_grads, n_grad = n_grad, n_neighbours = n_neighbours, aligned_grads = True, feat_selection = feat_selection, percentile = percentile)
n_sub = len(data)
n_features = data.shape[1] - 1
n_sub_feat_ratio = n_sub / n_features
scz_controls_ratio = len(data[data['diagnosis'] == 1]) / len(data[data['diagnosis'] == 0])
default_n_features = 499499
# Pycaret
exp = setup(data, target = 'diagnosis', session_id = 123)
best_model_obj = compare_models(fold = 10)
tuned_model = tune_model(best_model_obj, fold = 10)
predictions = predict_model(tuned_model)
test_metrics = pull()
best_model = test_metrics["Model"].values


# write results as a row in a csv
filename = 'results/ml_results.csv'

print("Writing results...")

# Add the results to the DataFrame
with open(filename, mode='a') as results_file:
    # Create a CSV writer object
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header row if the file is empty
    if results_file.tell() == 0:
        results_writer.writerow(["data_type", "N_grad", "comb_grads", "N_neighbours", "N_features", "N_sub", "N_sub/N_feat", "percent_feat", "feat_selection","best_model", "acc", "f1", "auc"])

    # Write the results for the current train and test session

    results_writer.writerow([data_type, n_grad, comb_grads, n_neighbours,n_features, n_sub, n_sub_feat_ratio, percentile, feat_selection, best_model, test_metrics['Accuracy'].values, test_metrics['F1'].values, test_metrics['AUC'].values])

print("Finished writing results.")


print("_____________________________MODEL FITTING FINISHED_____________________________")