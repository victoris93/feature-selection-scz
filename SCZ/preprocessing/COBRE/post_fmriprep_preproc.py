from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
from NeuroConn.gradient.gradient import get_gradients
import sys
import os

subject = sys.argv[1]
group = sys.argv[2]
data_path = f'/gpfs3/well/margulies/projects/data/COBRE/{group}'

dataset = FmriPreppedDataSet(data_path)
dataset.get_conn_matrix(subject, n_parcels = 1000, task = 'rest', save = True, output_space = 'MNI152NLin2009cAsym:res-2', concat_ts = True)
print(f"Connectivity matrix of subject {subject} saved.")
gradients = get_gradients(dataset, subject = subject, task = "rest", n_components = 1000, save = True)
print(f"Gradients for subject {subject} are saved.")
