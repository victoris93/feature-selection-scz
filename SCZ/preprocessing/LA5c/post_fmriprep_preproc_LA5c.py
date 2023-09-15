from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from NeuroConn.gradient.gradient import get_gradients
import sys
import os

subject =  sys.argv[1]
data_path = '/gpfs3/well/margulies/projects/data/LA5c'
wd = os.path.join(data_path, 'derivatives/fmriprep', f'sub-{subject}', 'func')

if os.path.exists(wd):
    LA5c_post_fmriprep = FmriPreppedDataSet(data_path)
    LA5c_post_fmriprep.get_conn_matrix(subject, task = 'rest', save = True, output_space = 'MNI152NLin2009cAsym:res-2')
    print(f"Connectivity matrix of subject {subject} saved.")
    gradients = get_gradients(LA5c_post_fmriprep, subject = subject, task = "rest", n_components = 1000, save = True)
    print(f"Connectivity matrix of subject {subject} saved: ", gradients.shape)
else:
    print(f"Func files of subject {subject} not found.")

