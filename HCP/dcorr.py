import numpy as np
from hyppo.independence import Dcorr
import os

subject_list = np.loadtxt('HCPSubjects.txt', dtype = int)
cluster_path = '/well/margulies/projects/data/hcp/schaefer1000'


grad1_ses1 = np.stack([np.load(f'{cluster_path}/{subject}/func/aligned_3gradients_ses1_{subject}_schaefer1000.npy')[:, 0] for subject in subject_list])
grad1_ses2 = np.stack([np.load(f'{cluster_path}/{subject}/func/aligned_3gradients_ses2_{subject}_schaefer1000.npy')[:, 0] for subject in subject_list])
dcorr_result = Dcorr().test(grad1_ses1, grad1_ses2)
