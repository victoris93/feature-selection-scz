import numpy as np
import nilearn
import pandas as pd
import os
import pycaret
from modeling_utils import *
from pycaret.classification import *

pca_features = np.load("group_pca_features.npy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pca_features = scaler.fit_transform(pca_features)

print("Input shape:", pca_features.shape)
participants = pd.read_csv("participants.csv")
pca_features["diagnosis"] = participants["diagnosis"].values

exp = setup(pca_features, target = 'diagnosis', session_id = 123)
lr = create_model('lr')
performance =pull()
performance.to_csv("results/pca_logreg_results.csv")

print("LogReg on pca features: results saved.")