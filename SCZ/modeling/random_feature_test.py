import sys
import os
import numpy as np
import pandas as pd
from modeling_utils import get_n_random_features
from pycaret.classification import *
import tqdm

model = sys.argv[1]
n_features = int(sys.argv[2])

data_csv = pd.read_csv("participants.csv")
features = np.load("all_features.npy")
features = pd.DataFrame(features)
for test in tqdm.tqdm(np.arange(1, 1001)):
    random_features = get_n_random_features(n_features, features)
    random_features["diagnosis"] = data_csv["diagnosis"].values

    experiment = setup(random_features, target = 'diagnosis', session_id = test, verbose=False)
    fit = create_model(model, verbose=False)

    performance = pull()
    perf =np.array((performance.loc["Mean"]["Accuracy"], performance.loc["Mean"]["AUC"]))
    del performance
    del fit
    del experiment

    if os.path.exists(f"results/null_dist/null_dist_{n_features}.npy"):
        null_dist = np.load(f"results/null_dist/null_dist_{n_features}.npy")
        null_dist = np.vstack((null_dist, perf))
        np.save(f"results/null_dist/null_dist_{n_features}.npy", null_dist)
        print(null_dist.shape)
    else:
        np.save(f"results/null_dist/null_dist_{n_features}.npy", perf)
        print("First values: ", perf.shape)

print(f"Random feature test for {n_features} features complete.")

