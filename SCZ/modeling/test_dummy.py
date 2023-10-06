import numpy as np
import pandas as pd
import os
import json
import sys
from modeling_utils import *
from pycaret.classification import *

output_csv = 'results/pca_dummy_test.csv'
pca_features = np.load("group_pca_features.npy")

models = json.load(open("models.json", "r"))
dummy_cl ='Dummy Classifier'
participants = pd.read_csv("participants.csv")
dummy_cl = models[dummy_cl]

pca_features = pd.DataFrame(pca_features)
pca_features["diagnosis"] = participants["diagnosis"].values
exp = setup(pca_features, target='diagnosis', session_id=123, fold_shuffle = True)
dummy_cl_train = create_model(dummy_cl)
perf = predict_model(dummy_cl_train)
perf = pull()

perf.to_csv(output_csv, index=False)
