import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import sys
import csv

region = int(sys.argv[1]) - 1

def perm_test_logreg(X_train, y_train, n_permutations=1000, region = region):
    # Initialize empty arrays to store beta coefficients and p-values for each region
    beta_coefficients = np.zeros(X_train.shape[1])
    
    # Fit a logistic regression model to the training data
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)

    # Compute the beta coefficients for the logistic regression model
    beta_coefficients = clf.coef_[0]

    # Compute the p-values for the beta coefficients using a permutation test
    np.random.seed(0)

    permuted_feature = np.random.permutation(X_train[:, region])
    permuted_X_train = np.copy(X_train)
    permuted_X_train[:, region] = permuted_feature

    clf_permuted = LogisticRegression(max_iter=10000)
    clf_permuted.fit(permuted_X_train, y_train)

    beta_coefficients_permuted = clf_permuted.coef_[0][region]
    beta_diff = beta_coefficients_permuted - beta_coefficients[region]
    
    permuted_beta_diffs = np.zeros(n_permutations)
    for j in range(n_permutations):
        permuted_y_train = np.random.permutation(y_train)
        clf_permuted_y = LogisticRegression(max_iter=10000)
        clf_permuted_y.fit(permuted_X_train, permuted_y_train)
        permuted_beta_coefficients = clf_permuted_y.coef_[0][region]
        permuted_beta_diffs[j] = permuted_beta_coefficients - beta_coefficients[region]
    
    p_value = np.sum(np.abs(permuted_beta_diffs) >= np.abs(beta_diff)) / n_permutations
    with open('results/logreg_perm_test_res.csv', mode='a', newline='') as csv_file:
        fieldnames = ['p_value', "beta", 'region']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:  # check if file is empty
            writer.writerow({'p_value': "p_value", "beta": "beta", 'region': "region"})
        writer.writerow({'p_value': p_value, 'beta': beta_coefficients[region], 'region': region + 1})


scz_path = '/well/margulies/projects/data/COBRE/clean_data/SCZ/schaefer1000'
controls_path = '/well/margulies/projects/data/COBRE/clean_data/controls/schaefer1000'

scz_subjects = np.loadtxt(f'{scz_path}/COBRE_SCZ_subjects.txt', dtype = str)
control_subjects = np.loadtxt(f'{controls_path}/COBRE_controls_subjects.txt', dtype = str)

# train sample
scz_subjects_train = scz_subjects[:60]
control_subjects_train = control_subjects[:60]
subjects_train = np.concatenate((control_subjects_train, scz_subjects_train))

group_train = np.zeros(len(scz_subjects_train) + len(control_subjects_train))
group_train[:len(scz_subjects_train)] = 1
group_train = group_train.astype(bool)
data_train = np.stack((subjects_train, group_train))

# test sample 
scz_subjects_test = scz_subjects[60:]
control_subjects_test = control_subjects[60:]
subjects_test = np.concatenate((scz_subjects_test, control_subjects_test))

group_test = np.zeros(len(scz_subjects_test) + len(control_subjects_test))
group_test[:len(scz_subjects_test)] = 1
group_test = group_test.astype(bool)
data_test = np.stack((subjects_test, group_test))

subject_df_train = pd.DataFrame(data_train.T, columns = ['Subject', 'Group'])
subject_df_test = pd.DataFrame(data_test.T, columns = ['Subject', 'Group'])

# load train & test dispersion
n_neighbours = 80

disp_train = []
for subject in scz_subjects_train:
    disp = np.load(f'{scz_path}/sub-{subject}/func/disp_6_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_train.append(disp)
for subject in control_subjects_train:
    disp = np.load(f'{controls_path}/sub-{subject}/func/disp_6_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_train.append(disp)

X_train_disp = np.stack(disp_train)

# prepare labels
y_train = subject_df_train['Group'].values

le = LabelEncoder()
y_train= le.fit_transform(y_train)

perm_test_logreg(X_train_disp, y_train)
print(f"Perm test for region {region+ 1} is done.")


