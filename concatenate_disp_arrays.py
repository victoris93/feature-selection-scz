import os, sys
import numpy as np

subjects = sorted(np.genfromtxt(r'/gpfs3/well/margulies/users/anw410/Vic/subj.txt'))
N_Neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
n_neighbours = N_Neighbours[int(sys.argv[1]) - 1]
odir=r'/gpfs3/well/margulies/users/anw410/Vic/Results' 

dispersion_output = []
for subject in subjects:
    """
    shape (sessions, vertices)
    """
    SubjDispersionArray = np.load("%s/dispersion_%s_%s_neighbours.npy" %(odir, subject, n_neighbours))
    dispersion_output.append(SubjDispersionArray)

np.save(arr = np.asarray(dispersion_output), file = "%s/dispersion_%s_neighbours.npy" %(odir, n_neighbours))