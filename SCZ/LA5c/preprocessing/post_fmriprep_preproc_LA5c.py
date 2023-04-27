from PyConn.preprocessing.preprocessing import FmriPreppedDataSet
from PyConn.gradient.gradient import get_gradients
import sys
import os

subject = sys.argv[1]
data_path = '/gpfs3/well/margulies/projects/data/LA5c'

wd = os.path.join(data_path, 'derivatives/fmriprep', f'sub-{subject}', 'func')
if os.path.exists(wd):
    LA5c_post_fmriprep = FmriPreppedDataSet(data_path)
    LA5c_post_fmriprep.clean_signal(subject, save = True)
    print(f"Clean ts of subject {subject} saved.")
    LA5c_post_fmriprep.get_conn_matrix(subject, save = True, subject_ts = f'{data_path}/derivatives/fmriprep/clean_data/sub-{subject}/func/clean-ts-sub-{subject}-rest-schaefer1000.npy')
    print(f"Connectivity matrix of subject {subject} saved.")
    gradients = get_gradients(data_path, subject = subject, task = "rest", n_components = 10, save = True)
else:
    print(f"Func files of subject {subject} not found.")
