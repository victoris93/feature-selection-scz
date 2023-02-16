import nibabel as nib
import brainspace
import numpy as np
from nilearn import signal
from brainspace.gradient import GradientMaps
import hcp_utils as hcp
from nilearn.interfaces.fmriprep import load_confounds_strategy

def get_corticalVertices(data):
	""" Get indices of Cortex Data from cifti file """
	cifti=nib.load(data)
	structMap=cifti.header.get_index_map(1)
	brainModels=list(structMap.brain_models)
	LCrtBM=brainModels[0]
	Lcrt_vrts=np.array(LCrtBM.vertex_indices)
	LnumVerts=LCrtBM.surface_number_of_vertices
	
	RCrtBM=brainModels[1]
	Rcrt_vrts=np.array(RCrtBM.vertex_indices)
	RnumVerts=RCrtBM.surface_number_of_vertices
	
	return {'lIDX':Lcrt_vrts,'lnverts':LnumVerts,'rIDX':Rcrt_vrts,'rnverts':RnumVerts}
	

clusterPath = "/well/margulies/projects/data/COBRE/derivatives/fmriprep/sub-A00000541"
subj_surf_ts_s1 = nib.load("%s/ses-20100101/func/sub-A00000541_ses-20100101_task-rest_space-fsLR_den-91k_bold.dtseries.nii" %clusterPath).get_fdata()
subj_surf_ts_s2 = nib.load("%s/ses-20110101/func/sub-A00000541_ses-20110101_task-rest_space-fsLR_den-91k_bold.dtseries.nii" %clusterPath).get_fdata()
confounds_out_s1 = load_confounds_strategy("%s/ses-20100101/func/sub-A00000541_ses-20100101_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" %clusterPath, denoise_strategy = "simple")
confounds_out_s2 = load_confounds_strategy("%s/ses-20110101/func/sub-A00000541_ses-20110101_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" %clusterPath, denoise_strategy = "simple")
confounds_out = np.concatenate([confounds_out_s1[0], confounds_out_s2[0]])

all_ts = np.concatenate([subj_surf_ts_s1, subj_surf_ts_s2])
clean_ts = signal.clean(all_ts, confounds = confounds_out)

cortex_clean_ts_labeled = np.array([hcp.cortex_data(i) for i in clean_ts])
mask = (cortex_clean_ts_labeled != 0).any(axis=0)
cortex_clean_ts = cortex_clean_ts_labeled[:, mask]
print("Shape of clean timeseries: ", cortex_clean_ts.shape)

correlation_matrix = np.corrcoef(cortex_clean_ts.T)
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
