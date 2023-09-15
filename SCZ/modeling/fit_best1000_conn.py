import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
from modeling_utils import *
from pycaret.classification import *

participants = pd.read_csv("participants.csv")
conn_feature_importance = np.load("results/z_feature_importance_matrix.npy")
conn_feature_importance = conn_feature_importance[:499500]

connectivity_features = np.load("z_all_features.npy")
connectivity_features = connectivity_features[:, :499500]
connectivity_features = pd.DataFrame(connectivity_features)

conn_labels = np.load("feature_labels.npy")[:499500]

best_conn_features = get_n_best_features(conn_feature_importance, 1000, connectivity_features, conn_labels)
best_conn_features["diagnosis"] = participants["diagnosis"]

del connectivity_features
del conn_feature_importance
del conn_labels

print("Fitting the models on 1000 best connectivity features...")
perfConn = fit_on_best_features(best_conn_features)
perfConn.to_csv("results/perfConn.csv")

print("Fitting done.")