import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
from modeling_utils import *
from pycaret.classification import *
import json

print("identifying the best model; train & test on 936 best connectivity features...")

models = json.load(open("models.json", "r"))
participants = pd.read_csv("participants.csv")
conn_feature_importance = np.load("results/importance_conn.npy")[0]

connectivity_features = np.load("all_features.npy")
connectivity_features = connectivity_features[:, :499500]
connectivity_features = pd.DataFrame(connectivity_features)

conn_labels = np.load("feature_labels.npy")[:499500]

best_conn_features = get_n_best_features(conn_feature_importance, 926, connectivity_features, conn_labels)
best_conn_features["diagnosis"] = participants["diagnosis"]
best_conn_features["age"] = participants["age"]
best_conn_features["sex"] = participants["sex"]
best_conn_features['sex'] = best_conn_features['sex'].replace({1: 'female', 0: 'male'})
best_conn_features["dataset"] = participants["dataset"]
best_conn_features["mean_fd"] = participants["mean_fd"] 

del connectivity_features
del conn_feature_importance
del conn_labels

print("Fitting the models on 936 best connectivity features...")
perfConn = fit_on_best_features(best_conn_features)
perfConn.to_csv("results/best_936_conn_cv.csv")
perfConn = perfConn.sort_values(by="Accuracy", ascending=False)
best_model = perfConn.iloc[0]["Model"]
best_model = models[best_model]
exp = setup(best_conn_features, target = "diagnosis", session_id = 123, normalize = True, categorical_features = ["sex", "dataset"], max_encoding_ohe = -1)
trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()
perf.to_csv("results/best_936_conn_test.csv", index=False)

print("Fitting done.")