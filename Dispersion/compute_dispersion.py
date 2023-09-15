from NeuroConn.gradient.dispersion import get_dispersion
from NeuroConn.preprocessing.preprocessing import FmriPreppedDataSet
import sys

n_grads = int(sys.argv[1])
n_neighbours = int(sys.argv[2])
sing = sys.argv[3]
if sing == 'sing':
    sing = True
else:
    sing = False

data_path = sys.argv[4]
fmriprepped_data = FmriPreppedDataSet(data_path)

for subject in fmriprepped_data.subjects:
    try:
        get_dispersion(fmriprepped_data, subject, n_grads, n_neighbours, task = 'rest', from_single_grads = sing, save = True, save_to = None, parcellation = 'schaefer', n_parcels = 1000,from_aligned_grads = True)
        print(f"Dispersion for subject {subject} from {data_path} computed.")
    except FileNotFoundError as e:
        print(f'Subject {subject} not found. Skipping.', e)
        continue
