from modeling_utils import fit_on_best_features
import sys
import os
import numpy as np
import pandas as pd

n_features = int(sys.argv[1])
best_features = pd.read_csv(f"best_features/{n_features}_best_features.csv")
diagnosis = pd.read_csv("participants.csv")['diagnosis'].values
best_features["diagnosis"] = diagnosis

performance = fit_on_best_features(best_features)
print(f"Models fitted for {n_features} features.")
performance.drop(columns = ["Recall", "Prec.", "Kappa", "MCC", "TT (Sec)"])
performance["N features"] = n_features

if os.path.exists("results/model_results.csv"):
    model_results = pd.read_csv("results/model_results.csv")
    model_results = pd.concat([model_results, performance], axis = 0)
    model_results.to_csv("results/model_results.csv", index = False)
else:
    performance.to_csv("results/model_results.csv", index = False)



