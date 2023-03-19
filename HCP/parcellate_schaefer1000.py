import numpy as np
import sys
import os
import brainspace
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.datasets import load_parcellation
import hcp_utils as hcp

subject = sys.argv[1]

cluster_path='/well/margulies/projects/data/hcp'
subject_output_dir = f'{cluster_path}/schaefer1000/{subject}/func'

if os.path.exists(f'{cluster_path}/{subject}/Rest/{subject}_concat_ts.npy'):
    schaefer1000 = load_parcellation("schaefer", 1000, join = True)

    subj_ts = np.load(f"{cluster_path}/{subject}/Rest/{subject}_concat_ts.npy")
    subj_ts_labeled = np.array([hcp.cortex_data(i) for i in subj_ts.T])

    subj_ts_schaefer1000 = reduce_by_labels(subj_ts_labeled, labels = schaefer1000)
    os.makedirs(subject_output_dir, exist_ok=True)

    np.save(f'{subject_output_dir}/{subject}_concat_ts_schaefer1000.npy', arr = subj_ts_schaefer1000)
    print("Parcellated timeseries saved.")
else:
    print("No timeseries found")