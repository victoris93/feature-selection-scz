import os
import sys
import pandas as pd

group = sys.argv[1]
data_path = f'/gpfs3/well/margulies/projects/data/COBRE/{group}'
participants = pd.read_csv(f"{data_path}/participants.tsv", sep='\t')   
participants = participants['participant_id'].tolist()
excluded_dir = os.path.join(data_path, 'derivatives', 'fmriprep', 'excluded')
if not os.path.exists(excluded_dir):
    os.mkdir(excluded_dir)

def get_sessions(subject_dir):
    subdirs = os.listdir(subject_dir)
    session_names = []
    for subdir in subdirs:
        if subdir.startswith('ses-'):
            session_names.append(subdir[4:])
    return session_names

for participant in participants:
    if participant.startswith('sub-'):
        participant = participant[4:]
    sub_dir = os.path.join(data_path, 'derivatives', 'fmriprep', f'sub-{participant}')
    try:
        sessions = get_sessions(sub_dir)
        sub_html = os.path.join(data_path, 'derivatives', 'fmriprep', f'sub-{participant}.html')
        confound_paths = []
        confound_list = []
        mean_fd_dict = {}
        for session_name in sessions:
            session_dir = os.path.join(sub_dir, f'ses-{session_name}', 'func')
            if os.path.exists(session_dir):
                confound_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('confounds_timeseries.tsv')]
                ses_conf = pd.concat([pd.read_csv(f, sep = '\t') for f in confound_files])
                mean_ses_fd = ses_conf['framewise_displacement'].mean()
                mean_fd_dict[session_name] = mean_ses_fd
        if all(fd > 0.5 for fd in list(mean_fd_dict.values())):
            os.system(f"mv {sub_dir} {excluded_dir}/sub-{participant}")
            os.system(f"mv {sub_html} {excluded_dir}/sub-{participant}.html")
            print(f"sub-{participant} from group {group} excluded: min session mean fd = {min(mean_fd_dict.values())}mm")
        else:
            for session in list(mean_fd_dict.keys()):
                if mean_fd_dict[session] > 0.5:
                    if not os.path.exists(f"{excluded_dir}/sub-{participant}"):
                        os.mkdir(f"{excluded_dir}/sub-{participant}")
                    os.system(f"mv {sub_dir}/ses-{session} {excluded_dir}/sub-{participant}/ses-{session}")
                    print(f"session {session} of sub-{participant} from group {group} excluded: mean fd = {mean_fd_dict[session]}mm")
    except FileNotFoundError:
        print(f"sub-{participant} from group {group} not found.")
        pass