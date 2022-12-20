import sys
import numpy as np 
import pandas as pd
import os
from pingouin import intraclass_corr

#N_Neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
#n_neighbours = N_Neighbours[int(sys.argv[1]) - 1]
odir = r'/gpfs3/well/margulies/users/anw410/Vic/Results'

inp = sys.argv[1]
n_neighbours = sys.argv[2]

if inp < 60:
    id_s = (inp-1)*1000
    id_e = inp * 1000 

else:
    id_s = (inp-1)*1000
    id_e = id_s + 412

icc_arr = np.empty((1000))

for i in range(id_s, id_e):
    ### ICC analysis ###
    Disp_df = pd.read_csv(f"{odir}/GradDispersionDf_{n_neighbours}Neighbours.csv")
    #Disp_df = pd.melt(Disp_df, id_vars = ("Vertex", "Session"), var_name = "Subject", value_name = "Dispersion")
    f = Disp_df.loc[[i,i + 59412]]
    f = f.drop(["Unnamed: 0","Vertex"], axis=1).melt(id_vars = ("Session"), var_name="Subject", value_name = "Dispersion")
    ICC = intraclass_corr(f, targets = "Subject", raters = "Session", ratings = "Dispersion")
    icc_arr[i] = ICC[ICC["Type"] == "ICC2"]["ICC"]

np.save(arr = icc_arr, file = f"{odir}/icc_arr_{n_neighbours}_neighbours_{inp}.npy")

