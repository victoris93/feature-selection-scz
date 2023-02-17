import nibabel as nib
import brainspace
import numpy as np
from nilearn import signal
from brainspace.gradient import GradientMaps
import hcp_utils as hcp
from nilearn.interfaces.fmriprep import load_confounds_strategy

cortex_indices = np.concatenate((hcp.vertex_info["grayl"], hcp.vertex_info["grayr"] + 32492))
smoothed_cln_ts = np.asarray(nib.load("smoothed/A00038624.010mm.z.dtseries.func.gii").agg_data())

print("Shape of timeseires: ", smoothed_cln_ts.shape)
smoothed_cln_ts = smoothed_cln_ts[:, cortex_indices]

#mask = (smoothed_cln_ts != 0).any(axis=0)
#cortex_clean_ts = cortex_clean_ts_labeled[:, mask]
#print("Shape of clean timeseries: ", cortex_clean_ts.shape)

correlation_matrix = np.corrcoef(smoothed_cln_ts.T)
np.save(arr = correlation_matrix, file = "output/corr_mat_A00038624.npy")
print("Corr matrix saved")

gm_pearson = GradientMaps(n_components=3, approach='pca', kernel='pearson')
gm_pearson.fit(correlation_matrix)

np.save(arr = np.asarray(gm_pearson.gradients_.T), file = "output/grads_A00038624_pearson.npy")
print("Pearson kernel gradients extracted successfully")
del gm_pearson

gm_cosine = GradientMaps(n_components=3, approach='pca', kernel='cosine')
gm_cosine.fit(correlation_matrix)
np.save(arr = np.asarray(gm_cosine.gradients_.T), file = "output/grads_A00038624_cosine.npy")
del gm_cosine
print("Cosine kernel gradients extracted successfully")
