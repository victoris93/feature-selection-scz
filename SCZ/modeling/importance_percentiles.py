import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
import tqdm
import sys

feature_type = sys.argv[1]
feature_type_dict = {"conn":"Connectivity", "grad":"Gradients", "disp":"Dispersion"}

feature_importance_conn = np.load("results/importance_conn.npy")[0]
#feature_importance_conn=(feature_importance_conn - np.min(feature_importance_conn)) / (np.max(feature_importance_conn) - np.min(feature_importance_conn))
feature_importance_grad = np.load("results/importance_grad.npy")[0]
#feature_importance_grad = (feature_importance_grad - np.min(feature_importance_grad)) / (np.max(feature_importance_grad) - np.min(feature_importance_grad))
feature_importance_centroid_disp = np.load("results/importance_centroid_disp.npy")[0]
feature_importance_cortex_disp = np.load("results/importance_cortex_disp.npy")[0]

feature_importance = np.concatenate((feature_importance_conn, feature_importance_grad, feature_importance_centroid_disp, feature_importance_cortex_disp))
# feature_importance = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance))
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