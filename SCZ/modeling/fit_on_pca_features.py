import numpy as np
import pandas as pd
import os
import pycaret
from modeling_utils import *
from pycaret.classification import *
import json

pca_features = np.load("group_pca_features.npy")
print("Input shape:", pca_features.shape)
pca_features = pd.DataFrame(pca_features)
participants = pd.read_csv("participants.csv")
pca_features["diagnosis"] = participants["diagnosis"].values

exp = setup(pca_features, target = 'diagnosis', fold_shuffle = True, session_id = 123)
models = compare_models()
performance = pull()
performance.to_csv("results/pca_cv_results.csv")

models = json.load(open("models.json", "r"))
performance = performance.sort_values(by="Accuracy", ascending=False)
best_model = performance.iloc[0]["Model"]
best_model = models[best_model]

# fit best model on confounds

trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()

perf.to_csv("results/pca_features_best_model_test.csv", index=False)

print("Model fits on pca features: results saved.")