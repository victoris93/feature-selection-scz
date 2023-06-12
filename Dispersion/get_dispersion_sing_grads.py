import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import pandas as pd

n_grads = int(sys.argv[1])
n_neighbours = int(sys.argv[2])

path_cobre_scz = '/gpfs3/well/margulies/projects/data/COBRE/SCZ/derivatives/fmriprep/clean_data'
path_cobre_controls = '/gpfs3/well/margulies/projects/data/COBRE/controls/derivatives/fmriprep/clean_data'

path_la5c = '/gpfs3/well/margulies/projects/data/LA5c/derivatives/fmriprep/clean_data'
path_ktt = '/gpfs3/well/margulies/projects/data/SRPBS_1600/KTT/derivatives/fmriprep/clean_data'

cobre_scz_subjects = pd.read_csv(f'{path_cobre_scz}/participants.tsv', sep= '\t')['participant_id'].values
cobre_controls_subjects = pd.read_csv(f'{path_cobre_controls}/participants.tsv', sep= '\t')['participant_id'].values
la5c_subjects = pd.read_csv(f'{path_la5c}/participants.tsv', sep= '\t')['participant_id'].values
ktt_subjects = pd.read_csv(f'{path_ktt}/participants.tsv', sep= '\t')['participant_id'].values

n_neighbours_list = np.arange(5, 505, 5)
n_grads_list = np.arange(1, 11)

for subject in cobre_scz_subjects:
    try:
        gradients = np.load(f'{path_cobre_scz}/sub-{subject}/func/aligned-10gradients-sub-{subject}-rest-schaefer1000.npy')[:,:, n_grads - 1].T
        hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
        distances, indices = hcp_ngbrs.kneighbors(gradients)
        subj_disp = distances.mean(axis = 1)
        np.save(f'{path_cobre_scz}/sub-{subject}/func/disp-sing{n_grads}grads-{n_neighbours}n-sub-{subject}-rest-schaefer1000.npy', subj_disp)
        print(f'Dispersion of gradient {n_grads} for {subject} saved.')
    except FileNotFoundError as e:
        print(f'Gradients not found for {subject}: {e}')
        continue

for subject in cobre_controls_subjects:
    try:
        gradients = np.load(f'{path_cobre_controls}/sub-{subject}/func/aligned-10gradients-sub-{subject}-rest-schaefer1000.npy')[:,:, n_grads - 1].T
        hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
        distances, indices = hcp_ngbrs.kneighbors(gradients)
        subj_disp = distances.mean(axis = 1)
        np.save(f'{path_cobre_controls}/sub-{subject}/func/disp-sing{n_grads}grads-{n_neighbours}n-sub-{subject}-rest-schaefer1000.npy', subj_disp)
        print(f'Dispersion of gradient {n_grads} for {subject} saved.')
    except FileNotFoundError as e:
        print(f'Gradients not found for {subject}: {e}')
        continue

for subject in la5c_subjects:
    try:
        gradients = np.load(f'{path_la5c}/{subject}/func/aligned-10gradients-{subject}-rest-schaefer1000.npy')[:,:, n_grads - 1].T
        hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
        distances, indices = hcp_ngbrs.kneighbors(gradients)
        subj_disp = distances.mean(axis = 1)
        np.save(f'{path_la5c}/{subject}/func/disp-sing{n_grads}grads-{n_neighbours}n-{subject}-rest-schaefer1000.npy', subj_disp)
        print(f'Dispersion of gradient {n_grads} for {subject} saved.')
    except FileNotFoundError as e:
        print(f'Gradients not found for {subject}: {e}')
        continue

for subject in ktt_subjects:
    try:
        gradients = np.load(f'{path_ktt}/{subject}/func/aligned-10gradients-{subject}-rest-schaefer1000.npy')[:,:, n_grads - 1].T
        hcp_ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
        distances, indices = hcp_ngbrs.kneighbors(gradients)
        subj_disp = distances.mean(axis = 1)
        np.save(f'{path_ktt}/{subject}/func/disp-sing{n_grads}grads-{n_neighbours}n-{subject}-rest-schaefer1000.npy', subj_disp)
        print(f'Dispersion of gradient {n_grads} for {subject} saved.')
    except FileNotFoundError as e:
        print(f'Gradients not found for {subject}: {e}')
        continue