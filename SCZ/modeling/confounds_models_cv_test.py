import numpy as np
import pandas as pd
import os
import json
import sys
from modeling_utils import *
from pycaret.classification import *

participants = pd.read_csv("participants.csv")
data = participants.drop(columns=['participant_id', 'path', 'Unnamed: 0'])
data["mean_fd"] = zscore(data["mean_fd"])
exp = setup(data, target = 'diagnosis', fold_shuffle = True, session_id = 123)
cv_models =compare_models()
cv_models = pull()
cv_models.to_csv("results/confound_model_results.csv")

models = json.load(open("models.json", "r"))

cv_models = cv_models.sort_values(by="Accuracy", ascending=False)
best_model = cv_models.iloc[0]["Model"]
best_model = models[best_model]

# fit best model on confounds

trained_model = create_model(best_model)
perf = predict_model(trained_model)
perf = pull()

perf.to_csv("results/confounds_best_model_test.csv", index=False)

print("Confounds model fit complete.")
