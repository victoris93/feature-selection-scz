import sys
sys.path.append('/gpfs3/well/margulies/users/cpy397/NeuroConn')
from NeuroConn.gradient.gradient import get_gradients
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import numpy as np
import os

subject =  sys.argv[1]
data_path = '/gpfs3/well/margulies/projects/data/LA5c'
wd = os.path.join(data_path, 'derivatives/fmriprep', "clean_data",  f'sub-{subject}', 'func')


if os.path.exists(wd):
    LA5c_post_fmriprep = FmriPreppedDataSet(data_path)
    gradients, eigenval = get_gradients(LA5c_post_fmriprep, subject = subject, task = "rest", n_components = 1000, save = False)
    np.save(wd + f'/sub-{subject}_task-rest_bold_gradients_eigenval.npy', eigenval)
else:
    print(f"Func files of subject {subject} not found.")

print("Eigenvalues saved.")