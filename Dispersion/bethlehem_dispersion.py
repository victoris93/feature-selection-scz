import numpy as np
from scipy.spatial.distance import euclidean
from nilearn.datasets import fetch_atlas_schaefer_2018
import sys
import os
sys.path.append('../../NeuroConn')

subject = sys.argv[1]
data_path = sys.argv[2]
yeo_schaefer_labels = {
    1 : "Vis",
    2 : "SomMot",
    3 : "DorsAttn",
    4 : "SalVentAttn",
    5 : "Limbic",
    6 : "Cont",
    7 : "Default"
}
schaefer_labels = np.array(fetch_atlas_schaefer_2018(1000, 7).labels).astype(str)

yeo_network_region_idx = []
for network in yeo_schaefer_labels.values():
    net_labels = np.array([i for i in schaefer_labels if network in i])
    region_idx = np.array([np.where(schaefer_labels == i)[0] for i in net_labels]).ravel()
    yeo_network_region_idx.append(region_idx)

subj_path = os.path.join(data_path, 'derivatives/fmriprep/clean_data', f"sub-{subject}", 'func')
grad_path = [i for i in os.listdir(subj_path) if 'gradients' in i and "aligned" in i][0]
grads = np.load(os.path.join(subj_path, grad_path))[0, :, :3]

within_net_disp = []
for network in yeo_network_region_idx:
    grad_network = grads[network]
    centroid = np.median(grad_network, axis=0)
    within_disp = np.array([euclidean(i, centroid)**2 for i in grad_network]).sum()
    within_net_disp.append(within_disp)
within_net_disp = np.array(within_net_disp)
np.save(os.path.join(subj_path, f"sub-{subject}_within_net_disp_bethlehem.npy"), within_disp)

between_net_disp = []
labels = []
for i, networkX in enumerate(yeo_network_region_idx):
    grad_network_X = grads[networkX]
    centroidX = np.median(grad_network_X, axis=0)
    remaining_networks = yeo_network_region_idx.copy()
    del remaining_networks[i]
    networkXYdisp = []
    labelsXY = []
    for networkY in remaining_networks:
        label = f'{i+1}_{yeo_network_region_idx.index(networkY)+1}'
        grad_network_Y = grads[networkY]
        centroidY = np.median(grad_network_Y, axis=0)
        between_disp = euclidean(centroidX, centroidY)
        networkXYdisp.append(between_disp)
        labelsXY.append(label)
    networkXYdisp_copy = networkXYdisp.copy()
    labelsXY_copy = labelsXY.copy()
    between_net_disp.append(networkXYdisp_copy)
    labels.append(labelsXY_copy)
between_net_disp = np.stack(between_net_disp)
unique_vals, indices = np.unique(between_net_disp, return_index=True)
labels = np.stack(labels).ravel()
labels = labels[indices]

np.save(os.path.join(subj_path, f"sub-{subject}_between_net_disp_bethlehem.npy"), between_net_disp)
np.save(os.path.join(subj_path, f"sub-{subject}_between_net_disp_bethlehem_labels.npy"), labels)

print(f"Within- & between-network computed for subject {subject}.")





