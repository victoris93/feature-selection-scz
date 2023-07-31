import sys
import os
import numpy as np
from modeling_utils import *

n_features = int(sys.argv[1])

best_fit = pd.read_csv("results/model_results.csv")
best_fit = best_fit[best_fit['N features'] == n_features]
dummy_acc = best_fit[best_fit["Model"] == "Dummy Classifier"]["Accuracy"].values[0]
dummy_auc = best_fit[best_fit["Model"] == "Dummy Classifier"]["AUC"].values[0]

best_fit = best_fit[best_fit['Accuracy'] > dummy_acc]
best_acc = best_fit["Accuracy"].mean()
best_fit = best_fit[best_fit["AUC"] > dummy_auc]
best_auc = best_fit["AUC"].mean()

data_csv = pd.read_csv("participants.csv")
features = np.load("all_features.npy")
features = pd.DataFrame(features)
features["diagnosis"] = data_csv["diagnosis"].values

features = get_n_random_features(n_features, features)
pval_acc, pval_auc = random_feature_test(n_features, features, best_acc, best_auc, data_csv, workers = -1, n_tests = 1000)
del features, data_csv

print(pval_acc, pval_auc)

model_results = pd.read_csv("results/model_results.csv")
model_results[model_results["N Features"] == n_features]["Pval Acc"] = pval_acc
model_results[model_results["N Features"] == n_features]["Pval AUC"] = pval_auc
model_results.to_csv("results/model_results.csv", index = False)

print(f"Random feature test for {n_features} features complete.")

