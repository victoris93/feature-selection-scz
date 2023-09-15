import numpy as np
import nilearn
import pandas as pd
import os
import pycaret

import json
from modeling_utils import *
from pycaret.classification import *


participants = pd.read_csv("participants.csv")
best_features = pd.read_csv("best_features/1000_best_features.csv")
best_features["diagnosis"] = participants["diagnosis"]

grad1_features = np.load("z_all_features.npy")[:, 499500:500500]
grad1_labels = np.load("feature_labels.npy")[499500:500500]
grad1_features = pd.DataFrame(grad1_features, columns = grad1_labels)
grad1_features["diagnosis"] = participants["diagnosis"]


perf1Grad = fit_on_best_features(grad1_features)
perfBest1000 = fit_on_best_features(best_features)

perf1Grad.to_csv("results/perf1Grad.csv")
perfBest1000.to_csv("results/perfBest1000.csv")

print("Fitting done.")

