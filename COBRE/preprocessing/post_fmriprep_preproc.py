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

# subject = sys.argv[1]

subject = 'A00018979'

data_path = '/gpfs3/well/margulies/projects/data/COBRE'
subject_list = np.loadtxt(f'{data_path}/COBRE_subjects.txt', dtype = 'str')

def get_sessions(subject, data = data_path):
    data = data + '/derivatives/fmriprep'
    subject_dir = f'{data}/sub-{subject}'
    subdirs = os.listdir(subject_dir)
    session_names = []
    for subdir in subdirs:
        if subdir.startswith('ses-'):
            session_names.append(subdir[4:])
    return session_names

def impute_nans(dataframe, pick_columns = None):
    imputer = SimpleImputer(strategy='mean')
    if pick_columns is not None and isinstance(pick_columns, (list, np.ndarray)):
        df_no_nans = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)[pick_columns]
    else:
        df_no_nans = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
    return df_no_nans

def get_confounds(subject, session_names, no_nans = True, pick_columns = None):
    confounds = []
    confounds_file_ses1 = f'{data_path}/derivatives/fmriprep/sub-{subject}/ses-{session_names[0]}/func/sub-{subject}_ses-{session_names[0]}_task-rest_desc-confounds_timeseries.tsv'
    confounds_out_s1 = pd.read_csv(confounds_file_ses1, sep = '\t')
    if no_nans == True:
        confounds_out_s1 = impute_nans(confounds_out_s1, pick_columns = pick_columns)
    confounds.append(confounds_out_s1)
    if len(session_names) > 1:
        confounds_file_ses2 = f'{data_path}/derivatives/fmriprep/sub-{subject}/ses-{session_names[1]}/func/sub-{subject}_ses-{session_names[1]}_task-rest_desc-confounds_timeseries.tsv'
        confounds_out_s2 = pd.read_csv(confounds_file_ses2, sep = '\t')
        if no_nans == True:
            confounds_out_s2 = impute_nans(confounds_out_s2, pick_columns = pick_columns)
        confounds.append(confounds_out_s2)
    else:
        confounds_out_s2 = None
    return confounds

def parcellate(subject_ts_paths, confounds, parcellation = 'schaefer'):
    parc_ts = []
    if parcellation == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir='/gpfs3/well/margulies/users/cpy397/nilearn_data', base_url= None, resume=True, verbose=1)
    masker =  NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, memory='nilearn_cache', verbose=5)
    parc_ts_s1 = masker.fit_transform(subject_ts_paths[0], confounds = confounds[0])
    print('First ses parcellated')
    parc_ts.append(parc_ts_s1)
    if len(subject_ts_paths) > 1:
        parc_ts_s2 = masker.fit_transform(subject_ts_paths[1], confounds = confounds[1])
        print('Both ses parcellated')
        parc_ts.append(parc_ts_s2)
    return parc_ts

def clean_signal(parc_ts_list):
    clean_ts = []
    clean_ts_s1 = signal.clean(parc_ts_list[0], t_r = 2, low_pass=0.08, high_pass=0.01, standardize=True, detrend=True)
    clean_ts.append(clean_ts_s1)
    if len(parc_ts_list) > 1:
        clean_ts_s2 = signal.clean(parc_ts_list[1], t_r = 2, low_pass=0.08, high_pass=0.01, standardize=True, detrend=True)
        clean_ts.append(clean_ts_s2)
    return clean_ts

session_names = get_sessions(subject)
picked_confounds = np.loadtxt('confounds.txt', dtype = 'str')
confounds = get_confounds(subject, session_names, pick_columns = picked_confounds)

subject_ts_paths = []

subj_ts_s1 = f'{data_path}/derivatives/fmriprep/sub-{subject}/ses-{session_names[0]}/func/sub-{subject}_ses-{session_names[0]}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
subject_ts_paths.append(subj_ts_s1)
if len(session_names) > 1:
    subj_ts_s2 = f'{data_path}/derivatives/fmriprep/sub-{subject}/ses-{session_names[1]}/func/sub-{subject}_ses-{session_names[1]}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    subject_ts_paths.append(subj_ts_s2)

parcellated_ts = parcellate(subject_ts_paths, confounds)
clean_parcellated_ts = clean_signal(parcellated_ts)
clean_parcellated_ts = np.stack(clean_parcellated_ts)
print('Shape of the timeseries: ', clean_parcellated_ts.shape)

output_dir = f'{data_path}/clean_data/sub-{subject}/func'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(file = f'{output_dir}/sub-{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-clean_bold.nii.gz', arr = clean_parcellated_ts)