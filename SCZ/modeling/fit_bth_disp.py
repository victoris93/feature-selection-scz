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

print("identifying the best model; train & test on Bethlehem dispersion...")

models = json.load(open("models.json", "r"))
participants = pd.read_csv("participants.csv")

bth_disp_features = np.load("all_features.npy")
bth_disp_features = bth_disp_features[:, 699500:699528]
bth_disp_features = pd.DataFrame(bth_disp_features)
bth_disp_labels = np.load("feature_labels.npy")[699500:699528]

bth_disp_features["diagnosis"] = participants["diagnosis"]
bth_disp_features["age"] = participants["age"]
bth_disp_features["sex"] = participants["sex"]
bth_disp_features['sex'] = bth_disp_features['sex'].replace({1: 'female', 0: 'male'})
bth_disp_features["dataset"] = participants["dataset"]
bth_disp_features["mean_fd"] = participants["mean_fd"]              

print("Fitting the models on centroid-based dispersion (28 features)...")
perf = fit_on_best_features(bth_disp_features)
perf.to_csv("results/bth_disp_cv.csv")
perf = perf.sort_values(by="Accuracy", ascending=False)
best_model = perf.iloc[0]["Model"]
best_model = models[best_model]
exp = setup(bth_disp_features, target = "diagnosis", session_id = 123, normalize = True, categorical_features = ["sex", "dataset"], max_encoding_ohe = -1)
trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()
perf.to_csv("results/bth_disp_test.csv", index=False)

print("Fitting done.")