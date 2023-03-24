import numpy as np
import sys
import os

subject_list = np.loadtxt("HCPSubjects.txt", dtype = str)
subject_indices = (int(sys.argv[1]), int(sys.argv[2]))
cluster_path='/well/margulies/projects/data/hcp/schaefer1000'
output_dir = "/well/margulies/users/cpy397/hcp/group_matrices"

for subject in subject_list:
    print(f"__________________START SUBJECT {subject}, TASK {list(subject_list).index(subject)}__________________")

    #subject_batch = f'{subject_list[subject_indices[0]]}_{subject_list[subject_indices[1]]}'

    subj_ses_ts_ses1 = np.load(f"{cluster_path}/{subject}/func/{subject}_concat_ts_schaefer1000.npy").T[:, :2400]
    subj_ses_ts_ses2 = np.load(f"{cluster_path}/{subject}/func/{subject}_concat_ts_schaefer1000.npy").T[:, 2400:]

    conn_mat_ses1 = np.corrcoef(subj_ses_ts_ses1)
    conn_mat_ses2 = np.corrcoef(subj_ses_ts_ses2)

    np.save(f'{cluster_path}/{subject}/func/conn_matrix_ses1_{subject}_schaefer1000', conn_mat_ses1)
    np.save(f'{cluster_path}/{subject}/func/conn_matrix_ses2_{subject}_schaefer1000', conn_mat_ses2)

    conn_mat_ses1_std =np.arctanh(conn_mat_ses1)
    conn_mat_ses2_std =np.arctanh(conn_mat_ses2)

    groupMatrixExistSes1 = os.path.exists(f'{output_dir}/group_matrix_{subject_batch}_ses1.npy')
    groupMatrixExistSes2 = os.path.exists(f'{output_dir}/group_matrix_{subject_batch}_ses2.npy')


    if not groupMatrixExistSes1:
        np.save(arr = conn_mat_ses1_std, file = f'{output_dir}/group_matrix_{subject_batch}_ses1.npy')
        print(f"first matrix of ses1 saved: subject {subject}")
    else:
        previous_sum_matrix = np.load(f'{output_dir}/group_matrix_{subject_batch}_ses1.npy')
        new_sum_matrix = np.add(previous_sum_matrix, conn_mat_ses1_std)
        np.save(arr = new_sum_matrix, file = f'{output_dir}/group_matrix_{subject_batch}_ses1.npy')
        print(f"matrix of subject {subject}, ses1 added")

    if not groupMatrixExistSes2:
        np.save(arr = conn_mat_ses2_std, file = f'{output_dir}/group_matrix_{subject_batch}_ses2.npy')
        print(f"first matrix of ses2 saved: subject {subject}")
    else:
        previous_sum_matrix = np.load(f'{output_dir}/group_matrix_{subject_batch}_ses2.npy')
        new_sum_matrix = np.add(previous_sum_matrix, conn_mat_ses2_std)
        np.save(arr = new_sum_matrix, file = f'{output_dir}/group_matrix_{subject_batch}_ses2.npy')
        print(f"matrix of subject {subject}, ses2 added")

    print(f"__________________END SUBJECT {subject}, TASK {list(subject_list).index(subject)}.__________________")

print(f"__________________SUMMED {len(subject_list[subject_indices[0]:subject_indices[1]+1])} SUBJECTS__________________")