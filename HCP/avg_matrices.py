import numpy as np
import os

group_sizes = [52, 52,52,52,52,52,52,53, 53, 54]
subject_files = os.listdir(os.getcwd())

subject_files.sort(key=lambda x: os.path.getmtime(x))
subject_files = [i.rsplit(".", 1)[0] for i in subject_files]

for index, group_size in enumerate(group_sizes):
    matrix_ses1 = np.load(f'{subject_files[2 * index]}.npy')
    matrix_avg_ses1 = matrix_ses1/group_size
    np.save(f'avg_{subject_files[2 * index]}.npy', matrix_avg_ses1)
    print(f"Avg matrices saved for group {index + 1}, ses 1")

    matrix_ses2 = np.load(f'{subject_files[2 * index + 1]}.npy')
    matrix_avg_ses2 = matrix_ses2/group_size
    np.save(f'avg_{subject_files[2 * index + 1]}.npy', matrix_avg_ses2)
    print(f"Avg matrices saved for group {index + 1}, ses 2")

print("Avg matrices saved")
