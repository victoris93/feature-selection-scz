from modeling_utils import fit_on_best_features
import sys
import os
import numpy as np
import pandas as pd

feature_type = sys.argv[1]
n_features = int(sys.argv[2])

best_features = pd.read_csv(f"best_features/{n_features}_best_features_{feature_type}.csv")
print("Features loaded...")
performance = fit_on_best_features(best_features)

print(f"CV done for all models for {n_features} features of {feature_type}.")
performance["N features"] = n_features

if os.path.exists(f"results/model_results_cv_{feature_type}.csv"):
    model_results = pd.read_csv(f"results/model_results_cv_{feature_type}.csv")
    model_results = pd.concat([model_results, performance], axis = 0)
    model_results.to_csv(f"results/model_results_cv_{feature_type}.csv", index = False)
else:
    performance.to_csv(f"results/model_results_cv_{feature_type}.csv", index = False)



