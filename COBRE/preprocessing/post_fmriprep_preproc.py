import nilearn
from nilearn import datasets
import numpy as np
import pandas as pd
import os
import sys
import nibabel as nib
from nilearn import signal
from sklearn.impute import SimpleImputer
from nilearn.maskers import NiftiLabelsMasker

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

def get_confounds(subject, session_names):
    confounds_file_ses1 = f'sub-{subject}/ses-{session_names[0]}/func/sub-{subject}_ses-{session_names[0]}_task-rest_desc-confounds_timeseries.tsv'
    confounds_out_s1 = pd.read_csv(confounds_file_ses1, sep = '\t')
    if len(session_names) > 1:
        confounds_file_ses2 = f'sub-{subject}/ses-{session_names[1]}/func/sub-{subject}_ses-{session_names[1]}_task-rest_desc-confounds_timeseries.tsv'
        confounds_out_s2 = pd.read_csv(confounds_file_ses2, sep = '\t')
    else:
        confounds_out_s2 = None
    return confounds_out_s1, confounds_out_s2

def parcellate(subject_paths, confounds, parcellation = 'schaefer'):
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)
    schaefer_masker =  NiftiLabelsMasker(labels_img=schaefer_atlas.maps, standardize=True, memory='nilearn_cache', verbose=5)
    clean_ts_s1 = schaefer_masker.fit_transform(subj_ts_s1, confounds = confounds_out_s1)
    clean_ts_s2 = schaefer_masker.fit_transform(subj_ts_s2, confounds = confounds_out_s2)
    

session_names = get_sessions(subject)

subj_ts_paths = []
subj_ts_s1 = f'sub-{subject}/ses-{session_names[0]}/func/sub-{subject}_ses-{session_names[0]}-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
subj_ts_paths.append(subj_ts_s1)
if session_names > 1:
    subj_ts_s2 = f'sub-{subject}/ses-{session_names[1]}/func/sub-{subject}_ses-{session_names[1]}-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    subj_ts_paths.append(subj_ts_s2)


confounds_file = 'sub-A00018979/ses-20100101/func/sub-A00018979_ses-20100101_task-rest_desc-confounds_timeseries.tsv'
confounds_out_s1 = pd.read_csv(confounds_file, sep = '\t')
confounds_file = 'sub-A00018979/ses-20110101/func/sub-A00018979_ses-20110101_task-rest_desc-confounds_timeseries.tsv'
confounds_out_s2 = pd.read_csv(confounds_file, sep = '\t')

picked_confounds = np.loadtxt('confounds.txt', dtype = 'str')

imputer = SimpleImputer(strategy='mean')
confounds_out_s1 = pd.DataFrame(imputer.fit_transform(confounds_out_s1), columns=confounds_out_s1.columns)[picked_confounds[1:]]
confounds_out_s2 = pd.DataFrame(imputer.fit_transform(confounds_out_s2), columns=confounds_out_s2.columns)[picked_confounds[1:]]

