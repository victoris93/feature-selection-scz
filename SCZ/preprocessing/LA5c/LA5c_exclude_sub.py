import os
import sys
import pandas as pd

data_path = f"/gpfs3/well/margulies/projects/data/LA5c"
participants = pd.read_csv(f"{data_path}/participants.tsv", sep='\t')   
participants = participants['participant_id'].tolist()
excluded_dir = os.path.join(data_path, 'derivatives', 'fmriprep', 'excluded')
if not os.path.exists(excluded_dir):
    os.mkdir(excluded_dir)
for participant in participants:
    if participant.startswith('sub-'):
        participant = participant[4:]
    sub_dir = os.path.join(data_path, 'derivatives', 'fmriprep', f'sub-{participant}')
    sub_html = os.path.join(data_path, 'derivatives', 'fmriprep', f'sub-{participant}.html')
    conf_path = os.path.join(sub_dir, 'func', f'sub-{participant}_task-rest_desc-confounds_timeseries.tsv')
    try:
        conf = pd.read_csv(conf_path, sep='\t')
        mean_fd = conf['framewise_displacement'].mean()
        if mean_fd > 0.5:
            os.system(f"mv {sub_dir} {excluded_dir}/sub-{participant}")
            os.system(f"mv {sub_html} {excluded_dir}/sub-{participant}.html")
            print(f"sub-{participant} excluded: mean fd = {mean_fd}mm")
    except FileNotFoundError:
        print(f"sub-{participant} not found")

