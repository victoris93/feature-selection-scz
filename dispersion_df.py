
from icc_utils import *
import os,sys
import numpy as np 
import pandas as pd

clusterPath = '/gpfs3/well/margulies/projects/data/hcpGrads'
# icc_utils should be here: '/well/margulies/projects/clinical_grads/Results'

subject = sys.argv[1]
odir = r'/gpfs3/users/margulies/anw410/v_work/Vic/Results'

#print(odir)
neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
grads_aligned = np.load(f"{clusterPath}/{subject}/{subject}.GradsAligned2Margulies2016.npy")

for n_neighbours in neighbours:
    dispersion = gradDispersion(grads_aligned[0], grads_aligned[1], n_neighbours = n_neighbours)
    dispersion_df_ses1 = make_measure_df(dispersion[0], "Disperison", subject, 1)
    dispersion_df_ses2 = make_measure_df(dispersion[1], "Disperison", subject, 2)
    dispersion_df = pd.concat((dispersion_df_ses1, dispersion_df_ses2))
    dispersion_df.to_csv(os.path.join(odir, 'dispersion_'+str(n_neighbours)+'_neighbours_'+str(subject)+'_df.csv'))
    

