import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
import tqdm
import sys

feature_type = sys.argv[1]
feature_type_dict = {"conn":"Connectivity", "grad":"Gradients", "disp":"Dispersion"}

feature_importance = np.load("results/feature_importance_matrix.npy")
feature_importance = np.max(feature_importance, axis = 1)
feature_importance = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance))
labels = np.load("feature_labels.npy")
labels = np.array([label.split("_")[0] for label in labels])

feature_importance_df = pd.DataFrame({"Feature Type": labels, "Importance": feature_importance})
feature_importance_df["Feature Type"] = feature_importance_df["Feature Type"].map({"conn":"Connectivity", "grad":"Gradients", "disp":"Dispersion"})

percentiles = []
feature_importance_df = feature_importance_df[feature_importance_df["Feature Type"] == feature_type_dict[feature_type]]
for score in tqdm.tqdm(feature_importance_df['Importance']):
    percentile = percentileofscore(feature_importance_df['Importance'], score)
    percentiles.append(percentile)
feature_importance_df['Percentile'] = percentiles

feature_importance_df.to_csv(f"results/{feature_type}_feature_importance.csv")
print("Percentile computed.")