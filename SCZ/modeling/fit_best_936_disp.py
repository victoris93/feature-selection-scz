import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
import sys
sys.path.append('../modeling_utils.py')
from modeling_utils import *
from pycaret.classification import *
import json

print("identifying the best model; train & test on 936 best dispersion features...")

models = json.load(open("models.json", "r"))
participants = pd.read_csv("participants.csv")
disp_feature_importance = np.load("results/feature_importance_cortex_disp.npy")

disp_features = np.load("all_features.npy")
disp_features = disp_features[:, 699528:]
disp_features = pd.DataFrame(disp_features)

disp_labels = np.load("feature_labels.npy")[699528:]

best_disp_features = get_n_best_features(disp_feature_importance, 926, disp_features, disp_labels)
best_disp_features["diagnosis"] = participants["diagnosis"]
best_disp_features["age"] = participants["age"]
best_disp_features["sex"] = participants["sex"]
best_disp_features['sex'] = best_disp_features['sex'].replace({1: 'female', 0: 'male'})
best_disp_features["dataset"] = participants["dataset"]
best_disp_features["mean_fd"] = participants["mean_fd"] 

del disp_features
del disp_feature_importance
del disp_labels

print("Fitting the models on 936 best dispersion (cortex-wide) features...")
perf = fit_on_best_features(best_disp_features)
perf.to_csv("results/best_936_disp_cv.csv")
perf = perf.sort_values(by="Accuracy", ascending=False)
best_model = perf.iloc[0]["Model"]
best_model = models[best_model]
exp = setup(best_disp_features, target = "diagnosis", session_id = 123, normalize = True, categorical_features = ["sex", "dataset"], max_encoding_ohe = -1)
trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()
perf.to_csv("results/best_936_disp_test.csv", index=False)

print("Fitting done.")