import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

n_grads = int(sys.argv[1])
n_neighbours = int(sys.argv[2])

scz_path = '/well/margulies/projects/data/COBRE/clean_data/SCZ/schaefer1000'
controls_path = '/well/margulies/projects/data/COBRE/clean_data/controls/schaefer1000'

scz_subjects = np.loadtxt(f'{scz_path}/COBRE_SCZ_subjects.txt', dtype = str)
control_subjects = np.loadtxt(f'{controls_path}/COBRE_controls_subjects.txt', dtype = str)

n_neighbours_list = np.arange(5, 505, 5)
n_grads_list = np.arange(1, 11)

for subject in scz_subjects:
    gradients = np.load(f'{scz_path}/sub-{subject}/func/aligned_10grads_{subject}_schaefer1000.npy')[:, :n_grads]
    hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
    distances, indices = hcp_ngbrs.kneighbors(gradients)
    subj_disp = distances.mean(axis = 1)
    np.save(f'{scz_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy', subj_disp)

for subject in control_subjects:
    gradients = np.load(f'{controls_path}/sub-{subject}/func/aligned_10grads_{subject}_schaefer1000.npy')[:, :n_grads]
    hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
    distances, indices = hcp_ngbrs.kneighbors(gradients)
    subj_disp = distances.mean(axis = 1)
    np.save(f'{controls_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy', subj_disp)
