from modeling_utils import get_n_best_features
import numpy as np
import pandas as pd
import os
import json
import sys

n_features = int(sys.argv[1])
feature_importance_matrix = np.load("results/z_feature_importance_matrix.npy")
all_features = np.load("z_all_features.npy")
all_features = pd.DataFrame(all_features)
feature_labels = np.load("feature_labels.npy")

best_features = get_n_best_features(feature_importance_matrix, n_features, all_features, feature_labels)
best_features.to_csv(f"best_features/{n_features}_best_features.csv", index = False)
print(best_features.head())

print(f"{n_features} best features saved.")