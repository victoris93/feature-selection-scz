from hcp_class_alignment import *
from icc_utils import *
import sys
import numpy as np 
import nibabel as nib 
import pandas as pd
import matplotlib.pyplot as plt
from surfplot.plotting import Plot
from sklearn.neighbors import NearestNeighbors
from pingouin import intraclass_corr

subj=sys.argv[1]
subj_inst=hcp_subj(subj,4)
labs_L_32k = nib.load('HCP/labels/fsLR.32k.L.label.gii').agg_data() # loading labels resampled with 2 diff. techniques
labs_R_32k = nib.load('HCP/labels/fsLR.32k.R.labelc.gii').agg_data()
labs_4k_metric  = np.concatenate((labs_L_32k, labs_R_32k))

### Get aligned gradients ###
grads_ses1_aligned = subj_inst.Gradsses1Aligned
grads_ses2_aligned = subj_inst.Gradsses2Aligned

neighbours = [50, 100, 200, 400, 800, 1600]
gradDisp_df_diff_neighbours = []
for n_neighbours in neighbours:
    gradDispAllSessions = gradDispersion(grads_ses1_aligned, grads_ses1_aligned)

    gradDisp_ses1 = gradDispAllSessions[0]
    gradDisp_ses2 = gradDispAllSessions[1]

    gradDisp_ses1_df = make_measure_df(gradDisp_ses1, "Dispersion", subjects = subj, session = 1)
    gradDisp_ses2_df = make_measure_df(gradDisp_ses2, "Dispersion", subjects = subj, session = 2)
    
    gradDisp_df = pd.concat([gradDisp_ses1_df, gradDisp_ses2_df])
    gradDisp_df["Neighbours"] = n_neighbours
    gradDisp_df_diff_neighbours.append(gradDisp_df)
gradDisp_df_diff_neighbours = pd.concat(gradDisp_df_diff_neighbours)
gradDisp_df_diff_neighbours.to_csv("gradDisp_df_diff_neighbours.csv")

### Compute gradient dispersion ###
#dispersion = gradDispersion(grads_ses1_aligned, grads_ses2_aligned, n_neighbours = 2000, save = True, file_name = "gradDispersion32k")
#d_ses1 = dispersion[0]
#vd_ses2 = dispersion[1]

#vd_ses1_df = make_measure_df(measure_array = vd_ses1, measure_name = "Dispersion", subjects = subj, session =1)
#vd_ses2_df = make_measure_df(measure_array = vd_ses2, measure_name = "Dispersion", subjects = subj, session =2)

#vd_df = pd.concat([vd_ses1_df, vd_ses2_df])
### Computing vertex-wise ICC ###

vertex_wise_ICC_diff_neighbours = []
for n_neighbours in neighbours:
    Disp_df = gradDisp_df_diff_neighbours[gradDisp_df_diff_neighbours["Neighbours"] == n_neighbours]
    vertex_wise_ICC = vertexICC(Disp_df, ICC_type = "ICC2")
    vertex_wise_ICC_diff_neighbours.append(vertex_wise_ICC)
vertex_wise_ICC_diff_neighbours = pd.concat(vertex_wise_ICC_diff_neighbours)

vertex_wise_ICC_diff_neighbours.to_csv("vertex_wise_ICC_diff_neighbours.csv")
