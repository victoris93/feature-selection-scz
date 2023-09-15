from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from NeuroConn.gradient.gradient import get_gradients
import sys
import os

subject =  sys.argv[1]
data_path = '/gpfs3/well/margulies/projects/data/SRPBS_1600/UTO'
wd = os.path.join(data_path, 'derivatives/fmriprep', f'sub-{subject}', 'func')

if os.path.exists(wd):
    post_fmriprep_dataset = FmriPreppedDataSet(data_path)
    post_fmriprep_dataset.get_conn_matrix(subject, task = 'rest', save = True, output_space = 'MNI152NLin2009cAsym:res-2')
    print(f"Connectivity matrix of subject {subject} saved.")
    gradients = get_gradients(post_fmriprep_dataset, subject = subject, task = "rest", n_components = 1000, save = True)
    print(f"Gradients of subject {subject} saved: ", gradients.shape)
    
else:
    print(f"Func files of subject {subject} not found.")
