import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def permutation_test(X_train, y_train, beta_coefficients, i, n_permutations):
    permuted_feature = np.random.permutation(X_train[:, i])
    permuted_X_train = np.copy(X_train)
    permuted_X_train[:, i] = permuted_feature

    clf_permuted = LogisticRegression(max_iter=10000)
    clf_permuted.fit(permuted_X_train, y_train)

    beta_coefficients_permuted = clf_permuted.coef_[0][i]
    beta_diff = beta_coefficients_permuted - beta_coefficients[i]

    permuted_beta_diffs = np.zeros(n_permutations)
    for j in range(n_permutations):
        permuted_y_train = np.random.permutation(y_train)
        clf_permuted_y = LogisticRegression(max_iter=10000)
        clf_permuted_y.fit(permuted_X_train, permuted_y_train)
        permuted_beta_coefficients = clf_permuted_y.coef_[0][i]
        permuted_beta_diffs[j] = permuted_beta_coefficients - beta_coefficients[i]

    p_value = np.sum(np.abs(permuted_beta_diffs) >= np.abs(beta_diff)) / n_permutations

    return p_value
