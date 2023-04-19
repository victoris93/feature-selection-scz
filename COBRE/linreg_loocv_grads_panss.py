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

COBRE_path = '/gpfs3/well/margulies/projects/data/COBRE'
scz_path = f'{COBRE_path}/clean_data/SCZ/schaefer1000'
controls_path = f'{COBRE_path}/clean_data/controls/schaefer1000'

scz_subjects = np.loadtxt(f'{scz_path}/COBRE_SCZ_subjects.txt', dtype = str)
control_subjects = np.loadtxt(f'{controls_path}/COBRE_controls_subjects.txt', dtype = str)
panss_total_file = 'panss_total.csv'

if panss_total_file not in os.listdir(f"{COBRE_path}/SCZ"):
    panss = pd.read_csv(f'{scz_path}/SCZ/COBRE_assessmentData_17938.csv')
    panss.replace({'Absent': 1, 'Minimal': 2, 'Mild': 3, 'MD':4, 'Moderate': 4, 'Moderate severe': 5, 'Severe': 6, 'Extreme': 7}, inplace=True)
    panss = panss.dropna(subset = ["question_value"])
    panss["question_value"] = panss["question_value"].astype(int)

    panss_total = pd.DataFrame(columns=['SubjectID', 'PANSS_Total'])

    for subject in scz_subjects:

        df_subject = panss[panss['subjectid'] == subject]
        total_score = df_subject['question_value'].sum()
        panss_total = panss_total.append({'SubjectID': subject, 'PANSS_Total': total_score}, ignore_index=True)
    panss_total.to_csv(f'{COBRE_path}/SCZ/panss_total.csv', index = False)
else:
    panss_total = pd.read_csv(f'{COBRE_path}/SCZ/panss_total.csv')

X_scz_grads_norm = []
for subject in scz_subjects:
    norm_grads_file = f'norm_aligned_10grads_{subject}_schaefer1000.npy'
    subject_path = f'{scz_path}/sub-{subject}/func'
    if norm_grads_file not in os.listdir(subject_path):
        scaler = StandardScaler()

        control_grads = []
        for control_subject in control_subjects:
            control_subject_path = f'{controls_path}/sub-{control_subject}/func'
            grads = np.load(f'{control_subject_path}/aligned_10grads_{control_subject}_schaefer1000.npy')
            control_grads.append(grads)
        control_grads = np.stack(control_grads)
        control_grads_reshaped = control_grads.reshape(control_grads.shape[0], -1)

        scz_grad = np.load(f'{subject_path}/aligned_10grads_{subject}_schaefer1000.npy')
        scz_grad_reshaped = np.expand_dims(scz_grad, axis = 0)
        scz_grad_reshaped = scz_grad_reshaped.reshape(scz_grad_reshaped.shape[0], -1)
        scaler.fit(control_grads_reshaped)
        scz_grad_norm = scaler.transform(scz_grad_reshaped)
        scz_grad_norm = scz_grad_norm.reshape(scz_grad.shape)
        np.save(f'{subject_path}/norm_aligned_10grads_{subject}_schaefer1000.npy', scz_grad_norm)
    else:
        scz_grad_norm = np.load(f'{subject_path}/norm_aligned_10grads_{subject}_schaefer1000.npy')
    X_scz_grads_norm.append(scz_grad_norm)
X_scz_grads_norm = np.stack(X_scz_grads_norm)

# Create a list of gradient combinations to try
grad_combinations = np.arange(1, 11)

# Initialize a DataFrame to store the results
linreg_df = pd.DataFrame(columns=["Num_Gradients", "MSE_Train", "MSE_Test", "Var_Explained"])

# Loop over the gradient combinations
for n_grads in grad_combinations:
    # Select the specified gradients for both control and patient data
    X_scz_grads_norm_sub = X_scz_grads_norm[:, :, :n_grads]
    y_scz = panss_total['PANSS_Total'].values

    # Initialize the regression model and LOOCV object
    reg = LinearRegression()
    loocv = LeaveOneOut()

    # Initialize lists to store the MSE for each fold
    mse_train = []
    mse_test = []
    var_explained = []

    # Loop over the LOOCV folds
    for train_idx, test_idx in loocv.split(X_scz_grads_norm_sub):

        # Split the data into training and testing sets
        X_train = X_scz_grads_norm_sub[train_idx]
        y_train = y_scz[train_idx]
        X_test = X_scz_grads_norm_sub[test_idx]
        y_test = y_scz[test_idx]

        # Fit the regression model to the training data
        reg.fit(X_train.reshape(len(train_idx), -1), y_train)

        # Predict the PNASS score for the training and testing data
        y_train_pred = reg.predict(X_train.reshape(len(train_idx), -1))
        y_test_pred = reg.predict(X_test.reshape(1, -1))

        # Calculate the MSE for the training and testing data
        mse_train.append(mean_squared_error(y_train, y_train_pred))
        mse_test.append(mean_squared_error(y_test, y_test_pred))
        var_explained.append(reg.score(X_train.reshape(len(train_idx), -1), y_train))

    # Compute the mean MSE across all folds for both training and testing data
    mean_mse_train = np.mean(mse_train)
    mean_mse_test = np.mean(mse_test)

    filename = 'linreg_loocv_grads_panss_perf.csv'
    # Add the results to the DataFrame
    with open(filename, mode='a') as results_file:
        # Create a CSV writer object
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write the header row if the file is empty
        if results_file.tell() == 0:
            results_writer.writerow(['Num_Gradients', 'MSE_Train', 'MSE_Test', 'Var_Explained'])

        # Write the results for the current train and test session
        results_writer.writerow([n_grads, mean_mse_train, mean_mse_test, var_explained])
