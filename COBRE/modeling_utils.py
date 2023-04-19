import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd

def fit_log_model(X, y, cv='loo'):

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    roc_auc_list = []
    deviance_list = []
    
    if cv == 'loo':
        cv = LeaveOneOut()


    for train_index, test_index in cv.split(X):
        clf = LogisticRegression(max_iter=10000)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        #roc_auc = roc_auc_score(y_test, y_prob[:, 1])

        log_likelihood_model = np.sum(y_test * np.log(y_prob[:, 1]) + (1 - y_test) * np.log(1 - y_prob[:, 1]))
        log_likelihood_saturated = np.sum(y_test * np.log(np.mean(y_test)) + (1 - y_test) * np.log(1 - np.mean(y_test)))
        deviance = -2 * (log_likelihood_model - log_likelihood_saturated)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        #roc_auc_list.append(roc_auc)
        deviance_list.append(deviance)
        

    perf_df = pd.DataFrame({
        'Accuracy': acc_list,
        'Precision': prec_list,
        'Recall': rec_list,
        'F1': f1_list,
        #'ROC_AUC': roc_auc_list,
        'Deviance': deviance_list
    })

    return perf_df


def plot_model_perf(perf_df, indep_var, metric, figsize, title = None, baseline = 0.5, show_min = False, show_max = True):
    '''
    Plots the performance of a model as a function of a single independent variable.
    perf_df: DataFrame containing the performance metrics for each value of the independent variable
    indep_var: Name of the independent variable
    figsize: Size of the figure (tuple)
    metric: Name of the performance metric to plot
    title: Title of the plot
    baseline: Baseline performance to plot
    show_min: If True, treats the minimum value of the performance metric as the optimal value and plots a vertical line at that point
    show_max: If True, treats the maximum value of the performance metric as the optimal value and plots a vertical line at that point
    '''

    fig, ax = plt.subplots(figsize=figsize)
    if not isinstance(indep_var, list):
        indep_var = [indep_var]
    if len(indep_var) > 1:
        if len(perf_df[indep_var[0]].unique()) > len(perf_df[indep_var[1]].unique()):
            hue_indep_var = indep_var[1]
            indep_var = indep_var[0]
        else:
            hue_indep_var = indep_var[0]
            indep_var = indep_var[1]
        grouped_data = perf_df.groupby([f'{hue_indep_var}', f'{indep_var}'])[f'{metric}'].agg(['mean']).reset_index()
        grouped_data.columns = [hue_indep_var, indep_var, f'Mean {metric}']
        for level in grouped_data[f'{hue_indep_var}'].unique():
            # Filter the data to only include rows with this N_grads value:
            subset = grouped_data[grouped_data[f'{hue_indep_var}'] == level]
            # Plot a line for this N_grads value:
            ax.plot(subset[f'{indep_var}'], subset[f'Mean {metric}'], label=f'{hue_indep_var}={level}')
        ax.set_xlabel(f'{indep_var}')
        ax.set_ylabel(f'Mean {metric}')
        ax.legend(loc='upper right')

    else:
        indep_var = indep_var[0]
        grouped_data = perf_df.groupby([indep_var]).agg({metric: ['mean', 'std']}).reset_index()
        grouped_data.columns = [indep_var, f'Mean {metric}', 'Standard Deviation']
        ax.errorbar(grouped_data[indep_var], grouped_data[f'Mean {metric}'], yerr=grouped_data['Standard Deviation'], fmt='o-', capsize=5)

    if show_min:
        optim_indep = grouped_data[indep_var][grouped_data[f'Mean {metric}'].idxmin()]
        best_metric = grouped_data[f'Mean {metric}'].min()
    elif show_max:
        optim_indep = grouped_data[indep_var][grouped_data[f'Mean {metric}'].idxmax()]
        best_metric = grouped_data[f'Mean {metric}'].max()
    if show_min or show_max:
        ax.axvline(optim_indep, color='red', linestyle='--')
        ax.text(optim_indep, ax.get_ylim()[0], f'{indep_var} = {optim_indep:.2f}', ha='center', va='bottom')
    
    ax.axhline(baseline, color='gray', linestyle='--')
    ax.text(0.05, 0.95, f"Best {metric} = {best_metric:.3f}", transform=ax.transAxes, fontsize=12, va='top')
    plt.title(f'{title}')
    plt.xlabel(indep_var)
    plt.ylabel(metric)

    