import nilearn
from nilearn import datasets
import numpy as np
import pandas as pd
import os
import sys
import nibabel as nib
from nilearn import signal
from sklearn.impute import SimpleImputer

subject = sys.argv[1]

data_path = "/well/margulies/projects/data/COBRE"
subject_list = np.loadtxt(f'{data_path}/COBRE_subjects.txt', dtype = 'str')

def get_sessions(subject, data = data_path):
    data = data + "/derivatives/fmriprep"
    subject_dir = f'{data}/sub-{subject}'
    subdirs = os.listdir(subject_dir)
    session_names = []
    for subdir in subdirs:
        if subdir.startswith("ses-"):
            session_names.append(subdir[4:])
    return session_names

session_names = get_sessions(subject)
if len(session_names) > 2:
    print(subject)

for subject in subject_list:
    session_names = get_sessions(subject)
    if len(session_names) == 2:
        print(subject)

confounds_file = 'sub-A00018979/ses-20100101/func/sub-A00018979_ses-20100101_task-rest_desc-confounds_timeseries.tsv'
confounds_out_s1 = pd.read_csv(confounds_file, sep = '\t')
confounds_file = 'sub-A00018979/ses-20110101/func/sub-A00018979_ses-20110101_task-rest_desc-confounds_timeseries.tsv'
confounds_out_s2 = pd.read_csv(confounds_file, sep = '\t')

picked_confounds = np.loadtxt('confounds.txt', dtype = 'str')

imputer = SimpleImputer(strategy='mean')
confounds_out_s1 = pd.DataFrame(imputer.fit_transform(confounds_out_s1), columns=confounds_out_s1.columns)[picked_confounds[1:]]
confounds_out_s2 = pd.DataFrame(imputer.fit_transform(confounds_out_s2), columns=confounds_out_s2.columns)[picked_confounds[1:]]

