import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import os
import json

all_data = np.load('/well/margulies/users/cpy397/SCZ/modeling/all_features.npy')
eigenvectors = np.zeros((all_data.shape))
n_components_conn = 302
n_components_gradients = 302
n_components_disp_bthlhm = 28
n_components_disp = 304

features_conn = all_data[:, :499500]
features_gradients = all_data[:, 499500:699500]
features_disp_bthlhm = all_data[:, 699500:699528]
features_disp = all_data[:, 699528:]
del all_data

pca = PCA(n_components = n_components_conn)
features_conn = pca.fit_transform(features_conn)
features_conn = zscore(features_conn, axis = 1)
print(f"PCA of connectivity: {features_conn.shape}")
eigenvectors_conn = pca.components_
eigenvectors_conn_minmax_feat = MinMaxScaler().fit_transform(eigenvectors_conn.T).T
eigenvectors_conn_minmax_comp = MinMaxScaler().fit_transform(eigenvectors_conn)
eigenvectors_conn = eigenvectors_conn_minmax_feat * eigenvectors_conn_minmax_comp
eigenvalues_conn = pca.explained_variance_

pca = PCA(n_components = n_components_gradients)
features_gradients = pca.fit_transform(features_gradients)
features_gradients = zscore(features_gradients, axis = 1)
print(f"PCA of gradients: {features_gradients.shape}")
eigenvectors_gradients = pca.components_
eigenvectors_gradients_minmax_feat = MinMaxScaler().fit_transform(eigenvectors_gradients.T).T
eigenvectors_gradients_minmax_comp = MinMaxScaler().fit_transform(eigenvectors_gradients)
eigenvectors_gradients = eigenvectors_gradients_minmax_feat * eigenvectors_gradients_minmax_comp
eigenvalues_gradients = pca.explained_variance_

pca = PCA(n_components = n_components_disp_bthlhm)
features_disp_bthlhm = pca.fit_transform(features_disp_bthlhm)
features_disp_bthlhm = zscore(features_disp_bthlhm, axis = 1)
print(f"PCA of within- and between-network dipsersion (Bethlehem): {features_disp_bthlhm.shape}")
eigenvectors_disp_bthlhm = pca.components_
eigenvectors_disp_bthlhm_minmax_feat = MinMaxScaler().fit_transform(eigenvectors_disp_bthlhm.T).T
eigenvectors_disp_bthlhm_minmax_comp = MinMaxScaler().fit_transform(eigenvectors_disp_bthlhm)
eigenvectors_disp_bthlhm = eigenvectors_disp_bthlhm_minmax_feat * eigenvectors_disp_bthlhm_minmax_comp
eigenvalues_disp_bthlhm = pca.explained_variance_

pca = PCA(n_components = n_components_disp)
features_disp = pca.fit_transform(features_disp)
features_disp = zscore(features_disp, axis = 1)
print(f"PCA of dispersion: {features_disp.shape}")
eigenvectors_disp = pca.components_
eigenvectors_disp_minmax_feat = MinMaxScaler().fit_transform(eigenvectors_disp.T).T
eigenvectors_disp_minmax_comp = MinMaxScaler().fit_transform(eigenvectors_disp)
eigenvectors_disp = eigenvectors_disp_minmax_feat * eigenvectors_disp_minmax_comp
eigenvalues_disp = pca.explained_variance_

pca_features = np.concatenate((features_conn, features_disp_bthlhm, features_gradients, features_disp), axis = 1)
eigenvectors[:302, :499500] = eigenvectors_conn
eigenvectors[302:604, 499500:699500] = eigenvectors_gradients
eigenvectors[604:632, 699500:699528] = eigenvectors_disp_bthlhm
eigenvectors[632:, 699528:] = eigenvectors_disp
eigenvalues = np.concatenate((eigenvalues_conn, eigenvalues_disp_bthlhm, eigenvalues_gradients, eigenvalues_disp), axis = 0)

print(f"PCA features: {eigenvectors.shape}")
print(f"PCA eigenvectors: {eigenvectors.shape}")
print(f"PCA eigenvalues: {eigenvalues.shape}")

np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_features.npy', pca_features)
np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_eigenvectors.npy', eigenvectors)
np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_eigenvalues.npy', eigenvalues)

print("Feature decomposition complete.")