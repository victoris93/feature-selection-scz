from modeling_utils import fit_on_best_features
import sys
import os
import numpy as np
import pandas as pd

n_features = int(sys.argv[1])
cv = sys.argv[2] # either "cv" or "test"


best_features = pd.read_csv(f"best_features/{n_features}_best_features.csv")
print("Features loaded...")
diagnosis = pd.read_csv("participants.csv")['diagnosis'].values
best_features["diagnosis"] = diagnosis

if cv == "cv":
    performance = fit_on_best_features(best_features, cv = True)
else:
    performance = fit_on_best_features(best_features, cv = False)

print(f"Models fitted for {n_features} features.")
performance.drop(columns = ["Recall", "Prec.", "Kappa", "MCC", "TT (Sec)"])
performance["N features"] = n_features
print(performance)

if os.path.exists(f"results/model_results_{cv}.csv"):
    model_results = pd.read_csv(f"results/model_results_{cv}.csv")
    model_results = pd.concat([model_results, performance], axis = 0)
    model_results.to_csv(f"results/model_results_{cv}.csv", index = False)
else:
    performance.to_csv(f"results/model_results_{cv}.csv", index = False)



