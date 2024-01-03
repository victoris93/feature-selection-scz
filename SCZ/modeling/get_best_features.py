import sys
sys.path.append('../modeling_utils.py')
from modeling_utils import get_n_best_features
import numpy as np
import pandas as pd
import os
import json
import sys

feature_type = sys.argv[1]
n_features = int(sys.argv[2])

participants = pd.read_csv("participants.csv")

all_features = np.load("all_features.npy")
feature_labels = np.load('feature_labels.npy')
if feature_type == "conn":
    features = all_features[:, :499500]
    labels = feature_labels[:499500]
elif feature_type == "grad":
    features = all_features[:, 499500:699500]
    labels = feature_labels[499500:699500]
elif feature_type == "cortex_disp":
    features = all_features[:, 699528:]
    labels = feature_labels[699528:]
features = pd.DataFrame(features, columns = labels)

if not os.path.exists(f"best_features/{n_features}_best_features_{feature_type}.csv"):
    feature_importance = np.load(f"results/importance_{feature_type}.npy")
    best_features = get_n_best_features(feature_importance, n_features, features, labels)
    best_features["diagnosis"] = participants["diagnosis"]
    best_features["age"] = participants["age"]
    best_features["sex"] = participants["sex"]
    best_features['sex'] = best_features['sex'].replace({1: 'female', 0: 'male'})
    best_features["dataset"] = participants["dataset"]
    best_features["mean_fd"] = participants["mean_fd"]

    best_features.to_csv(f"best_features/{n_features}_best_features_{feature_type}.csv", index = False)
    print(best_features.head())

    print(f"{n_features} best features saved.")
else:
    print(f"{n_features} best features already exist. Exiting...")
    sys.exit()
