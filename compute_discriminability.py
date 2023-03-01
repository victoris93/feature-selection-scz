import os, sys
import numpy as np
from hyppo.discrim import DiscrimOneSample

#N_Neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
n_neighbours = sys.argv[1]

#subjects = np.loadtxt("SubjectsCompleteData.txt", dtype= str)[:500]
#ses_1_labs = np.stack((subjects, np.zeros(500)), axis = 1)
#ses_2_labs = np.stack((subjects, np.ones(500)), axis = 1)
#ses_labels = np.concatenate((ses_1_labs, ses_2_labs))

ses_labels = np.concatenate([np.zeros(1018), np.ones(1018)], axis=0)

odir=r'/gpfs3/well/margulies/users/cpy397/DispersionResults'

csv_file = odir + "/Discriminability/discrim_%s.csv" % n_neighbours

disp = np.load(odir + "/DispersionAllSubj_%s_neighbours.npy" % n_neighbours)
result = DiscrimOneSample().test(disp, ses_labels, workers=-1)
stat = result.stat
p_value = result.pvalue
np.save(arr = result.null_dist, file = odir + "/Discriminability/null_distribution_%s_neighbours.npy" % n_neighbours)

with open(csv_file, "w") as f:
    f.write("n_neighbours,stat,p_value\n")
    f.write("%s,%s,%s\n" % (n_neighbours, stat, p_value))
    f.flush()
