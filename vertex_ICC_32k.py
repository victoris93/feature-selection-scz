from icc_utils import vertexICC
import sys
import numpy as np 
import pandas as pd
import os

n_neighbours = sys.argv[1]

odir = r'/gpfs3/well/margulies/users/anw410/Vic/Results'

### ICC analysis ###
Disp_df = pd.read_csv(f"{odir}/GradDispersionDf_{n_neighbours}Neighbours.csv")
vertexICC(Disp_df, ICC_type = "ICC2", save = True, filename = f"{odir}/{n_neighbours}_Neighbours_DispVertexICC.csv")

