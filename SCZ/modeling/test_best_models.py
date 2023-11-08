import numpy as np
import pandas as pd
import os
import json
import sys
from modeling_utils import *
from pycaret.classification import *

feature_type = sys.argv[1]
n_features = int(sys.argv[2])
print(f"Train and test of the best model for {n_features} of {feature_type}...")

output_csv = f'results/best_models_test_{feature_type}.csv'
if os.path.exists(output_csv):
    best_model_tests = pd.read_csv(output_csv)
    if n_features in best_model_tests["N features"].values:
        print(f"Best model already tested for {n_features} features. Exiting...")
        sys.exit()

models = json.load(open("models.json", "r"))

cv_results = pd.read_csv(f'results/model_results_cv_{feature_type}.csv')
cv_results = cv_results[cv_results["N features"] == n_features]
cv_results = cv_results.sort_values(by="Accuracy", ascending=False)
best_model = cv_results.iloc[0]["Model"]
best_model = models[best_model]

# fit best model on full dataset
best_features = pd.read_csv(f'best_features/{n_features}_best_features_{feature_type}.csv')
exp = setup(best_features, target='diagnosis', session_id=123, verbose=True, fold_shuffle = True, normalize = True, categorical_features = ["sex", "dataset"], max_encoding_ohe = -1)
trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()
perf["N features"] = n_features

if not os.path.exists(output_csv):
    perf.to_csv(output_csv, index=False)
else:
    performance = pd.read_csv(output_csv)
    performance = pd.concat([performance, perf])
    performance.to_csv(output_csv, index=False)

# cv_results_pca = pd.read_csv('pca_cv_results.csv')
# cv_results_confounds = pd.read_csv('confound_model_results.csv')

print(f"Test performance for {n_features} of {feature_type} saved")