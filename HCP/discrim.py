import numpy as np
from hyppo.discrim import DiscrimOneSample
import os

ses_labels = np.concatenate([np.zeros(525), np.ones(525)], axis=0)
subject_list = np.loadtxt('HCPSubjects.txt', dtype = int)

subject_gradient_files = [file for file in os.listdir(os.getcwd()) if file.startswith("dispersion_")]
subject_gradient_files = [i.rsplit(".", 1)[0] for i in subject_gradient_files]
dispersion = np.stack([np.load(f'{i}.npy') for i in subject_gradient_files])

cluster_path = '/well/margulies/projects/data/hcp/schaefer1000'

disp_ses1 = np.stack([np.load(f'{cluster_path}/{subject}/func/dispersion_ses1_{subject}_schaefer1000.npy') for subject in subject_list])
disp_ses2 = np.stack([np.load(f'{cluster_path}/{subject}/func/dispersion_ses2_{subject}_schaefer1000.npy') for subject in subject_list])


disp = np.row_stack((disp_ses1, disp_ses2))
result = DiscrimOneSample().test(disp, ses_labels, workers=-1)
result
