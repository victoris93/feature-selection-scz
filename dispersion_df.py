
from icc_utils import *
import sys
import numpy as np 
import pandas as pd

clusterPath = '/well/margulies/projects/clinical_grads/Results'
# all utils should be here: '/well/margulies/projects/clinical_grads/Results'

subject = sys.argv[1]
n_neighbours = sys.argv[2]
odir=sys.argv[3]
grads_aligned = np.load(f"{clusterPath}/{subject}.GradsAligned2Margulies2016.npy")

dispersion = gradDispersion(grads_aligned[0], grads_aligned[1], n_neighbours = n_neighbours, save = True, filename = f"{subject}_dispersion.npy")

dispersion_df_ses1 = make_measure_df(dispersion[0], "Disperison", subject, 1, True, f"{subject}_dispersion_df_ses1.csv")
dispersion_df_ses2 = make_measure_df(dispersion[1], "Disperison", subject, 2, True, f"{subject}_dispersion_df_ses2.csv")
dispersion_df = pd.concat((dispersion_df_ses1, dispersion_df_ses2))
dispersion_df.to_csv(f"{subject}_dispersion_df.csv")