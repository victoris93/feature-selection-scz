import pandas as pd
import os
import sys
import csv
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import modeling_utils as mu

n_grads = sys.argv[1]
n_neighbours = sys.argv[2]

COBRE_path = '/gpfs3/well/margulies/projects/data/COBRE'
scz_path = f'{COBRE_path}/clean_data/SCZ/schaefer1000'
controls_path = f'{COBRE_path}/clean_data/controls/schaefer1000'

scz_subjects = np.loadtxt(f'{scz_path}/COBRE_SCZ_subjects.txt', dtype = str)
control_subjects = np.loadtxt(f'{controls_path}/COBRE_controls_subjects.txt', dtype = str)

y = np.concatenate([np.ones(len(scz_subjects)), np.zeros(len(control_subjects))]).astype(int)
log_disp_single_grads = []

disp_array = []
for subject in scz_subjects:
    disp = np.load(f'{scz_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_array.append(disp)
for subject in control_subjects:
    disp = np.load(f'{controls_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_array.append(disp)
disp_array = np.stack(disp_array)

print(f"Fitting model on gradient {n_grads} with {n_neighbours} neighbours...")

result_df = mu.fit_log_model(disp_array, y)
result_df['N_Grads'] = n_grads
result_df['N_Neighbours'] = n_neighbours
result_df.to_csv(f'logreg_loocv_disp_{n_grads}_{n_neighbours}n.csv')

print("Finished writing results.")
