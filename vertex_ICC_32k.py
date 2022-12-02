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

os.system("mkdir clinical_grads")
os.system("mkdir clinical_grads/results")

parser = argparse.ArgumentParser(description='Embeds functional Connectivity Matrix using PCA and diffusion map embedding, and then determines how far peaks are across sessions of the same subject',\
	usage='vertex_ICC_32k.py --subj <subject list> --odir <output directory> ',\
	epilog=("Example usage: "+"vertex_ICC_32k.py --subj subj_list.txt --odir /well/margulies/projects/clinical_grads"),add_help=True)


if len(sys.argv) < 2:
	parser.print_help()
	sys.exit(1)

req_grp = parser.add_argument_group(title='Required arguments')
req_grp.add_argument('--subj',type=str,metavar='',required=True,help='HCP list of subjects')
req_grp.add_argument('--odir',type=str,metavar='',required=True,help='Output directory base. Output will be saved as odir/subj/...')
req_grp.add_argument('--kernel',type=str,metavar='',required=True,help='specify smoothing kernel of time series')

op_grp= parser.add_argument_group(title='Optional arguments')
args=parser.parse_args()

subj_file=args.subj
odir=args.odir
kernel=args.kernel

clusterPath='/well/margulies/projects/data/hcpGrads'
subj=sys.argv[1]
subjects = np.loadtxt(subj_file, dtype = "str")

hcp_subj_list = []
for subject in subjects:
    hcp_subj_list.append(hcp_subj(subject,4))
    
### Get aligned gradients ###

grads_ses1_aligned = np.stack([subject.Gradsses1Aligned for subject in hcp_subj_list])
grads_ses2_aligned = np.stack([subject.Gradsses2Aligned for subject in hcp_subj_list])

neighbours = [50, 100, 200, 400, 800, 1600]
gradDisp_df_diff_neighbours = []
for n_neighbours in neighbours:
    gradDispAllSessions = gradDispersion(grads_ses1_aligned, grads_ses1_aligned, save = True, filename = "Results/gradDispersion_%sneighbours.npy" % n_neighbours)

    gradDisp_ses1 = gradDispAllSessions[0]
    gradDisp_ses2 = gradDispAllSessions[1]

    gradDisp_ses1_df = make_measure_df(gradDisp_ses1, "Dispersion", subjects = subjects, session = 1)
    gradDisp_ses2_df = make_measure_df(gradDisp_ses2, "Dispersion", subjects = subjects, session = 2)
    
    gradDisp_df = pd.concat([gradDisp_ses1_df, gradDisp_ses2_df])
    gradDisp_df["Neighbours"] = n_neighbours
    gradDisp_df_diff_neighbours.append(gradDisp_df)
gradDisp_df_diff_neighbours = pd.concat(gradDisp_df_diff_neighbours)
gradDisp_df_diff_neighbours.to_csv(clusterPath + "Results/gradDisp_df_diff_neighbours.csv")

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

vertex_wise_ICC_diff_neighbours.to_csv(clusterPath + "vertex_wise_ICC_diff_neighbours.csv")
