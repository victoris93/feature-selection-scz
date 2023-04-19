import pandas as pd
import numpy as np
import csv
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

scz_path = '/well/margulies/projects/data/COBRE/clean_data/SCZ/schaefer1000'
controls_path = '/well/margulies/projects/data/COBRE/clean_data/controls/schaefer1000'

scz_subjects = np.loadtxt(f'{scz_path}/COBRE_SCZ_subjects.txt', dtype = str)
control_subjects = np.loadtxt(f'{controls_path}/COBRE_controls_subjects.txt', dtype = str)

linreg_df = pd.DataFrame(columns=["N_grads", "N_Neighbors", "MSE_Train", "MSE_Test", "MAE_Train", "MAE_Test", "R2_Train", "R2_Test", "Obs_Left_Out", "Actual_Score", "Predicted_Score"])

n_grads = sys.argv[1]
n_neighbours = sys.argv[2]


disp_array = []
for subject in scz_subjects:
    disp = np.load(f'{scz_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_array.append(disp)
for subject in control_subjects:
    disp = np.load(f'{controls_path}/sub-{subject}/func/disp_{n_grads}_{n_neighbours}n_{subject}_schaefer1000.npy')
    disp_array.append(disp)

disp_array = np.stack(disp_array)


# Normalize scz disp relative to controls
scaler = StandardScaler()
scaler.fit(disp_array[len(scz_subjects):])
X_scz_disp_norm = scaler.transform(disp_array[:len(scz_subjects)])
disp_array[:len(scz_subjects)] = X_scz_disp_norm

# Select the subset of subjects with the desired n_neighbours
panss_total = pd.read_csv('COBRE_PANSS_Total.csv', index_col=0)
X_scz_disp_norm_sub = disp_array[:len(scz_subjects)]
y_scz = panss_total['PANSS_Total'].values

# Initialize the LOOCV object
loocv = LeaveOneOut()

# Initialize lists to store the MSE for each fold
mse_train = []
mse_test = []
mae_train = []
mae_test = []
obs_left_out = []
actual_scores = []
predicted_scores_test = []
predicted_scores_train = []

# Loop over the LOOCV folds
print("Fitting model...")
for train_idx, test_idx in loocv.split(X_scz_disp_norm_sub):
    reg = LinearRegression()

    # Split the data into training and testing sets
    X_train = X_scz_disp_norm_sub[train_idx]
    y_train = y_scz[train_idx]
    X_test = X_scz_disp_norm_sub[test_idx]
    y_test = y_scz[test_idx]

    # Fit the regression model to the training data
    reg.fit(X_train.reshape(len(train_idx), -1), y_train)

    # Predict the PNASS score for the training and testing data
    y_train_pred = reg.predict(X_train.reshape(len(train_idx), -1))
    y_test_pred = reg.predict(X_test.reshape(1, -1))

    # Calculate the MSE, MAE, vscode-webview://0s8407ja4lfle12o90208dp55topvghfobgfe41snmpsvsq5k2bk/2576395a-4341-442b-95d4-b87661af61d5and R^2 for the training and testing data
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    mae_train.append(mean_absolute_error(y_train, y_train_pred))
    mae_test.append(mean_absolute_error(y_test, y_test_pred))
    
    # Append the index of the observation left out and the actual and predicted scores
    obs_left_out.append(test_idx[0])
    actual_scores.append(y_test[0])
    predicted_scores_test.append(y_test_pred[0])
    predicted_scores_train.append(y_train_pred[0])

r2_train = r2_score(actual_scores, predicted_scores_train)
r2_test = r2_score(actual_scores, predicted_scores_test)
# Compute the mean MSE, MAE, and R^2 across all folds for both training and testing data
mean_mse_train = np.mean(mse_train)
mean_mse_test = np.mean(mse_test)
mean_mae_train = np.mean(mae_train)
mean_mae_test = np.mean(mae_test)


filename = 'results/linreg_loocv_disp_panss_perf.csv'

print("Writing results...")

# Add the results to the DataFrame
with open(filename, mode='a') as results_file:
    # Create a CSV writer object
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header row if the file is empty
    if results_file.tell() == 0:
        results_writer.writerow(["N_grads", "N_Neighbors", "MSE_Train", "MSE_Test", "MAE_Train", "MAE_Test", "R2_Train", "R2_Test", "Obs_Left_Out", "Actual_Score", "Predicted_Score"])

    # Write the results for the current train and test session
    for i in range(len(mse_train)):
        results_writer.writerow([n_grads, n_neighbours, mse_train[i], mse_test[i], mae_train[i], mae_test[i], r2_train, r2_test, obs_left_out[i], actual_scores[i], predicted_scores_test[i]])

print("Finished writing results.")


