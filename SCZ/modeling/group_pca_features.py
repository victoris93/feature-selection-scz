import numpy as np
from sklearn.decomposition import PCA
import os
import json

all_data = np.load('/well/margulies/users/cpy397/SCZ/modeling/all_features.npy')[:, :699500]
eigenvectors = np.zeros((all_data.shape))
n_components = int(all_data.shape[0]/2)
features_conn = all_data[:, :499500]
features_gradients = all_data[:, 499500:699500]
del all_data


pca = PCA(n_components = n_components)
features_conn = pca.fit_transform(features_conn)
print(f"PCA of connectivity: {features_conn.shape}")
eigenvectors_conn = pca.components_
eigenvalues_conn = pca.explained_variance_

pca = PCA(n_components = n_components)
features_gradients = pca.fit_transform(features_gradients)
print(f"PCA of gradients: {features_gradients.shape}")
eigenvectors_gradients = pca.components_
eigenvalues_gradients = pca.explained_variance_

# pca = PCA(n_components = n_components)
# features_disp = pca.fit_transform(features_disp)
# print(f"PCA of dispersion: {features_disp.shape}")
# eigenvectors_disp = pca.components_
# eigenvalues_disp = pca.explained_variance_

pca_features = np.concatenate((features_conn, features_gradients), axis = 1)
eigenvectors[:n_components, :499500] = eigenvectors_conn
eigenvectors[n_components:, 499500:699500] = eigenvectors_gradients

eigenvalues = np.concatenate((eigenvalues_conn, eigenvalues_gradients), axis = 0)

print(f"PCA features: {eigenvectors.shape}")
print(f"PCA eigenvectors: {eigenvectors.shape}")
print(f"PCA eigenvalues: {eigenvalues.shape}")

np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_features_nodisp.npy', pca_features)
np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_eigenvectors_nodisp.npy', eigenvectors)
np.save('/well/margulies/users/cpy397/SCZ/modeling/group_pca_eigenvalues_nodisp.npy', eigenvalues)

print("Feature decomposition complete.")