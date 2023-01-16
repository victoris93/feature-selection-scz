import os, sys
import numpy as np
import pandas as pd
from hyppo.discrim import DiscrimOneSample

N_Neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
ses_labels = np.concatenate([np.zeros(500), np.ones(500)], axis=0)
odir=r'/gpfs3/well/margulies/users/cpy397/DispersionResults'
discriminability = []

for n_neighbours in N_Neighbours:
    disp = np.load(odir + "/DispersionAllSubj_%s_neighbours.npy" % n_neighbours)
    result = DiscrimOneSample().test(disp, ses_labels)
    result_array = np.array([n_neighbours, result.stat, result.pvalue])
    discriminability.append(result_array)
discriminability = np.asarray(discriminability)
discr_df = pd.DataFrame(discriminability, columns = ["N_neighbours", "Discriminability", "p-value"])
discr_df.to_csv(odir + "/Discriminability/discrim.csv")
