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

data_path = '/gpfs3/well/margulies/projects/data/COBRE'

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

def get_confounds(subject, no_nans = True, pick_columns = None, data_dir = data_path):
    confound_paths = []
    confound_list = []
    session_names = get_sessions(subject)
    subject_dir = os.path.join(data_dir, 'derivatives', 'fmriprep', f'sub-{subject}')
    for session_name in session_names:
        session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
        if os.path.exists(session_dir):
            confound_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('confounds_timeseries.tsv')]
            confound_paths.extend(confound_files)
    if no_nans == True:
        for confounds_path in confound_paths:
            confounds = pd.read_csv(confounds_path, sep = '\t')
            confounds = impute_nans(confounds, pick_columns = pick_columns)
            confound_list.append(confounds)
    else:
        for confounds_path in confound_paths:
            confounds = pd.read_csv(confounds_path, sep = '\t')
            confound_list.append(confounds)
    return confound_list

def parcellate(subject_ts_paths, confounds, parcellation = 'schaefer', gsr = False):
    parc_ts_list = []
    if parcellation == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir='/gpfs3/well/margulies/users/cpy397/nilearn_data', base_url= None, resume=True, verbose=1)
    masker =  NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, memory='nilearn_cache', verbose=5)
    for subject_ts, subject_confounds in zip(subject_ts_paths, confounds):
        if gsr == False:
            parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds.drop("global_signal", axis = 1))
            parc_ts_list.append(parc_ts)
        else:
            parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds)
            parc_ts_list.append(parc_ts)
    return parc_ts_list

def clean_signal(parc_ts_list):
    clean_ts = []
    for parc_ts in parc_ts_list:
        clean_ts_s1 = signal.clean(parc_ts, t_r = 2, low_pass=0.08, high_pass=0.01, standardize=True, detrend=True)
        clean_ts.append(clean_ts_s1[10:]) # discarding first 10 volumes
    return clean_ts

def get_ts_paths(subject, data_dir = data_path):
    subject_dir = f'{data_dir}/derivatives/fmriprep/sub-{subject}'
    ts_paths = []
    session_names = get_sessions(subject)
    for session_name in session_names:
        session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
        if os.path.exists(session_dir):
            session_run_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')]
            ts_paths.extend(session_run_files)
    return ts_paths

session_names = get_sessions(subject)
picked_confounds = np.loadtxt('confounds.txt', dtype = 'str')
confounds = get_confounds(subject, pick_columns = picked_confounds)
subject_ts_paths = get_ts_paths(subject)

parcellated_ts = parcellate(subject_ts_paths, confounds)
clean_parcellated_ts = clean_signal(parcellated_ts)
clean_parcellated_ts = np.stack(clean_parcellated_ts)
print('Shape of the timeseries: ', clean_parcellated_ts.shape)

output_dir = f'{data_path}/clean_data/sub-{subject}/func'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(file = f'{output_dir}/sub-{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-clean_bold', arr = clean_parcellated_ts)


def get_confounds(no_nans = True):
    if no_nans == True:
        print("No nans is true")