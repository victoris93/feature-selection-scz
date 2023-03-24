import numpy as np
import brainspace
from brainspace.datasets import load_parcellation
from brainspace.gradient import GradientMaps
import os

subject_matrix_files = [file for file in os.listdir(os.getcwd()) if file.startswith("avg_group")]
subject_matrix_files = [i.rsplit(".", 1)[0] for i in subject_matrix_files]
prefix = "avg_group_matrix_"
subject_prefices = [matrix_file.replace(prefix, "") for matrix_file in subject_matrix_files]

for matrix_file, subject_prefix in zip(subject_matrix_files, subject_prefices):
    gm = GradientMaps(n_components=3, kernel = "cosine", approach= 'pca')
    matrix = np.load(f'{matrix_file}' + ".npy")
    if np.isnan(matrix).any():
        nan_indices = np.where(np.isnan(matrix))
        matrix[nan_indices] = .0000000001
    if np.isinf(matrix).any():
        inf_indices = np.where(np.isinf(matrix))
        matrix[inf_indices] = 1
    gm.fit(matrix)
    gradients = gm.gradients_
    np.save(f'group_gradient_{subject_prefix}', gradients)

print(f"__________________GRADIENTS {subject_prefix} SAVED__________________")


subject_list = np.loadtxt("HCPSubjects.txt", dtype = str)
cluster_path='/well/margulies/projects/data/hcp/schaefer1000'
output_dir = "/well/margulies/users/cpy397/hcp/group_matrices"

for subject in subject_list:
    matrix_ses1_path = f'{cluster_path}/{subject}/func/conn_matrix_ses1_{subject}_schaefer1000.npy'
    matrix_ses2_path = f'{cluster_path}/{subject}/func/conn_matrix_ses2_{subject}_schaefer1000.npy'
    matrix_ses1 = np.load(matrix_ses1_path)
    matrix_ses2 = np.load(matrix_ses2_path)
    matrices = [matrix_ses1, matrix_ses2]
    for index, matrix in enumerate(matrices):
        gm = GradientMaps(n_components=3, kernel = "cosine", approach= 'pca')
        if np.isnan(matrix).any():
            nan_indices = np.where(np.isnan(matrix))
            matrix[nan_indices] = .0000000001
        if np.isinf(matrix).any():
            inf_indices = np.where(np.isinf(matrix))
            matrix[inf_indices] = 1
        gm.fit(matrix)
        gradients = gm.gradients_
        np.save(f'{cluster_path}/{subject}/func/3gradients_ses{index + 1}_{subject}_schaefer1000', gradients)