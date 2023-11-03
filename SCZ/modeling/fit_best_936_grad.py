import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
import json
from modeling_utils import *
from pycaret.classification import *

print("identifying the best model; train & test on 936 best gradient values...")

models = json.load(open("models.json", "r"))
participants = pd.read_csv("participants.csv")
grad_feature_importance = np.load("results/importance_grad.npy")[0]

grad_features = np.load("all_features.npy")
grad_features = grad_features[:, 499500:699500]
grad_features = pd.DataFrame(grad_features)

grad_labels = np.load("feature_labels.npy")[499500:699500]

best_grad_features = get_n_best_features(grad_feature_importance, 926, grad_features, grad_labels)
best_grad_features["diagnosis"] = participants["diagnosis"]
best_grad_features["age"] = participants["age"]
best_grad_features["sex"] = participants["sex"]
best_grad_features['sex'] = best_grad_features['sex'].replace({1: 'female', 0: 'male'})
best_grad_features["dataset"] = participants["dataset"]
best_grad_features["mean_fd"] = participants["mean_fd"]                          


del grad_features
del grad_feature_importance
del grad_labels

print("Fitting the models on 936 best gradient values...")
perf = fit_on_best_features(best_grad_features)
perf.to_csv("results/best_936_grad_cv.csv")
perf = perf.sort_values(by="Accuracy", ascending=False)
best_model = perf.iloc[0]["Model"]
best_model = models[best_model]
exp = setup(best_grad_features, target = "diagnosis", session_id = 123, normalize = True, categorical_features = ["sex", "dataset"], max_encoding_ohe = -1)
trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()
perf.to_csv("results/best_936_grad_test.csv", index=False)

print("Fitting done.")



