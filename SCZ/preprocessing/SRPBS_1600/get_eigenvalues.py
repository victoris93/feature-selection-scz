import sys
sys.path.append('/gpfs3/well/margulies/users/cpy397/NeuroConn')
from NeuroConn.gradient.gradient import get_gradients
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import numpy as np
import os

subject = sys.argv[1]
site = sys.argv[2]
data_path = f'/gpfs3/well/margulies/projects/data/SRPBS_1600/{site}'
wd = os.path.join(data_path, 'derivatives/fmriprep','clean_data', f'sub-{subject}', 'func')

if os.path.exists(wd):
    dataset = FmriPreppedDataSet(data_path)
    gradients, eigenval = get_gradients(dataset, subject = subject, task = "rest", n_components = 1000, save = False, aligned = False)
    np.save(wd + f'/sub-{subject}_task-rest_bold_gradients_eigenval.npy', eigenval)
else:
    print(f"Func files of subject {subject} not found.")

print("Eigenvalues saved.")