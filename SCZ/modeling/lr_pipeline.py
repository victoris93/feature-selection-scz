import numpy as np
import pandas as pd
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from modeling_utils import permutation_importance_pca

participants = pd.read_csv("participants.csv")
all_features = np.load("all_features.npy")
feature_labels = np.load('feature_labels.npy')
all_features = pd.DataFrame(all_features, columns = feature_labels)

conn_cols = all_features.iloc[:, :499500].columns
grad_cols = all_features.iloc[:, 499500:699500].columns
centroid_disp_cols = all_features.iloc[:, 699500:699528].columns
cortex_disp_cols = all_features.iloc[:, 699528:].columns

all_features["dataset"] = participants["dataset"]
all_features["sex"] = participants["sex"]
all_features['sex'] = all_features['sex'].replace({1: 'female', 0: 'male'})
all_features["age"] = participants["age"]
all_features["mean_fd"] = participants["mean_fd"]
diagnosis = participants["diagnosis"] # target

X_train, X_test, y_train, y_test = train_test_split(all_features, diagnosis, test_size=0.25, random_state=42)

pca_conn = Pipeline(
    steps = [("group_whiten", StandardScaler()),
             ('pca', PCA(n_components = 0.2)),
            ("pca_whiten", StandardScaler())]
)

pca_grad = Pipeline(
    steps = [("group_whiten", StandardScaler()),
             ('pca', PCA(n_components = 0.2)),
            ("pca_whiten", StandardScaler())]
)

pca_centroid_disp = Pipeline(
    steps = [("group_whiten", StandardScaler()),
             ('pca', PCA(n_components = 28)),
            ("pca_whiten", StandardScaler())]
)

pca_cortex_disp = Pipeline(
    steps = [("group_whiten", StandardScaler()),
             ('pca', PCA(n_components = 0.2)),
            ("pca_whiten", StandardScaler())]
)

cat_encoder = Pipeline(
    steps = [("cat_encoder", OneHotEncoder(handle_unknown="ignore"))]
)
whiten = Pipeline(
    steps = [("whiten", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("pca_conn", pca_conn, conn_cols),
        ("pca_grad", pca_grad, grad_cols),
        ("pca_centroid_disp", pca_centroid_disp, centroid_disp_cols),
        ("pca_cortex_disp", pca_cortex_disp, cortex_disp_cols),
        ("encode_dataset", cat_encoder, ["dataset"]),
        ("encode_sex", cat_encoder, ["sex"]),
        ("whiten_fd", whiten, ["mean_fd"]),
        ("whiten_age", whiten, ["age"])
    ]
)

# CV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter = 10000)
clf = Pipeline([('preprocessor', preprocessor),
                ('lr',lr)])

print("Starting 10-fold CV")
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
cv_results = cross_validate(clf, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'], return_estimator =True, n_jobs = -1)
cv_scores_df = pd.DataFrame({
    'fold': range(1, cv.get_n_splits() + 1),
    'accuracy': cv_results['test_accuracy'],
    'f1_score': cv_results['test_f1']
})

from sklearn.metrics import accuracy_score, f1_score
print("Assessing test performance...")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)
f1_score_test = f1_score(y_test, y_pred)
test_scores_df = pd.DataFrame({
    'fold': ['test'],
    'accuracy': [accuracy_test],
    'f1_score': [f1_score_test]
})

all_scores_df = pd.concat([cv_scores_df, test_scores_df])
all_scores_df.to_csv("results/pca_logreg_results.csv", index = False)

# Permutation importance 

lr = LogisticRegression(random_state=42, max_iter = 10000)
# clf = Pipeline([('preprocessor', preprocessor),
#                 ('lr',lr)])
X_train_transformed = preprocessor.fit_transform(X_train)

print(f"N_components for connectivity: {preprocessor.named_transformers_['pca_conn'].named_steps['pca'].n_components_}")
print(f"N_components for gradient: {preprocessor.named_transformers_['pca_grad'].named_steps['pca'].n_components_}")
print(f"N_components for centroid_disp: {preprocessor.named_transformers_['pca_centroid_disp'].named_steps['pca'].n_components_}")
print(f"N_components for cortex_disp: {preprocessor.named_transformers_['pca_cortex_disp'].named_steps['pca'].n_components_}")

X_test_transformed = preprocessor.transform(X_test)
lr_trained = lr.fit(X_train_transformed, y_train)

print("Computing permutation feature importance...")
mean_importances_pca, std_importances_pca, importances_pca = permutation_importance_pca(lr, X_test_transformed, y_test, n_repeats=10000)


np.save("results/mean_importance_pca.npy", mean_importances_pca)
np.save("results/std_importances_pca.npy", std_importances_pca)
np.save("results/importances_pca.npy", importances_pca)


coefs_latent_conn = mean_importances_pca[preprocessor.output_indices_['pca_conn']]
coefs_conn = preprocessor.named_transformers_["pca_conn"].inverse_transform(np.array([coefs_latent_conn]))

coefs_latent_grad = mean_importances_pca[preprocessor.output_indices_['pca_grad']]
coefs_grad = preprocessor.named_transformers_['pca_grad'].inverse_transform(np.array([coefs_latent_grad]))

coefs_latent_cortex_disp = mean_importances_pca[preprocessor.output_indices_['pca_cortex_disp']]
coefs_cortex_disp = preprocessor.named_transformers_['pca_cortex_disp'].inverse_transform(np.array([coefs_latent_cortex_disp]))

coefs_latent_centroid_disp = mean_importances_pca[preprocessor.output_indices_['pca_centroid_disp']]
coefs_centroid_disp = preprocessor.named_transformers_['pca_centroid_disp'].inverse_transform(np.array([coefs_latent_centroid_disp]))

np.save("results/importance_conn.npy", coefs_conn[0])
np.save("results/importance_grad.npy", coefs_grad[0])
np.save("results/importance_cortex_disp.npy", coefs_cortex_disp[0])
np.save("results/importance_centroid_disp.npy", coefs_centroid_disp[0])



print("Feature importance computed.")


# results = cross_validate(clf, all_features, participants["diagnosis"].values, cv=10, scoring=['accuracy', 'f1'], return_train_score=True)

# print("Starting permutation importance...")

# from sklearn.inspection import permutation_importance

# trained_logreg = clf.fit(X_train, y_train)
# trained_logreg.score(X_test, y_test)

# perm_acc = permutation_importance(trained_logreg, X_test, y_test,n_repeats=100, random_state=42, n_jobs = -1)
# perm_sorted_idx = perm_acc.importances_mean
# perm_std = perm_acc.importances_std
# null_dist_coefs = perm_acc.importances

# np.save("results/null_dist_coefs.npy", null_dist_coefs)
# np.save("results/perm_coef_std.npy", perm_std)
# np.save("results/perm_mean_coef_sorted.npy", perm_sorted_idx)

# print("Permutation importance done.")
