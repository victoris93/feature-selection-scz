from hcp_class_alignment import *
import argparse
from icc_utils import *
import sys
import numpy as np 
import nibabel as nib 
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from pingouin import intraclass_corr

# all utils should be here: '/well/margulies/projects/clinical_grads/Results'

grad_array = sys.argv[1]
odir=sys.argv[2]

### Get aligned gradients ###

hcp_aligned_grad_list = list(np.load("{clusterPath}/aligned_grads.npy"))


grads_ses1_aligned = np.stack([subject[0] for subject in hcp_aligned_grad_list])
grads_ses2_aligned = np.stack([subject[1] for subject in hcp_aligned_grad_list])

### Create DFs for ICC analysis ###

neighbours = [50, 100, 200, 400, 800, 1600]
gradDisp_df_diff_neighbours = []
for n_neighbours in neighbours:
    gradDispAllSessions = gradDispersion(grads_ses1_aligned, grads_ses1_aligned, save = True, filename = "{odir}/gradDispersion_{n_neighbours}neighbours.npy")

    gradDisp_ses1 = gradDispAllSessions[0]
    gradDisp_ses2 = gradDispAllSessions[1]

    gradDisp_ses1_df = make_measure_df(gradDisp_ses1, "Dispersion", subjects = subjects, session = 1)
    gradDisp_ses2_df = make_measure_df(gradDisp_ses2, "Dispersion", subjects = subjects, session = 2)
    
    gradDisp_df = pd.concat([gradDisp_ses1_df, gradDisp_ses2_df])
    gradDisp_df["Neighbours"] = n_neighbours
    gradDisp_df_diff_neighbours.append(gradDisp_df)
gradDisp_df_diff_neighbours = pd.concat(gradDisp_df_diff_neighbours)
gradDisp_df_diff_neighbours.to_csv("{odir}/gradDisp_df_diff_neighbours.csv")

### ICC analysis ###

vertex_wise_ICC_diff_neighbours = []
for n_neighbours in neighbours:
    Disp_df = gradDisp_df_diff_neighbours[gradDisp_df_diff_neighbours["Neighbours"] == n_neighbours]
    vertex_wise_ICC = vertexICC(Disp_df, ICC_type = "ICC2")
    vertex_wise_ICC_diff_neighbours.append(vertex_wise_ICC)
vertex_wise_ICC_diff_neighbours = pd.concat(vertex_wise_ICC_diff_neighbours)

vertex_wise_ICC_diff_neighbours.to_csv("{odir}/vertex_wise_ICC_diff_neighbours.csv")
