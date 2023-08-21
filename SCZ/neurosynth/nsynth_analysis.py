import os
import sys
import pandas as pd
import nibabel as nb
import nibabel as nib
import numpy as np
import nilearn
from nilearn import datasets
import nimare
from nimare.dataset import Dataset
from nimare.decode import discrete
sys.path.append('../modeling')
from modeling_utils import get_feature_regions
import re

n_features = sys.argv[1]
n_features_list = np.loadtxt("n_features.txt", dtype=str).tolist()

neurosynth_dset = Dataset.load("neurosynth_dataset.pkl.gz")

schaefer_map = nilearn.datasets.fetch_atlas_schaefer_2018(1000, yeo_networks = 7)
schaefer_map = nb.load(schaefer_map['maps'])

best_features = pd.read_csv(f"../modeling/best_features/{n_features}_best_features.csv")
regions_kept = get_feature_regions(best_features, False)
regions_kept = np.where(regions_kept != 0)[0] + 1
if n_features != "100":
    prev_best_features = pd.read_csv(f"../modeling/best_features/{n_features_list[n_features_list.index(n_features) - 1]}_best_features.csv")
    prev_regions_kept = get_feature_regions(prev_best_features, False)
    prev_regions_kept = np.where(prev_regions_kept != 0)[0] + 1
    regions_kept = np.setdiff1d(regions_kept, prev_regions_kept)

roi = np.zeros_like(schaefer_map.get_fdata())
print("Number of regions kept: ", len(regions_kept))

for region in regions_kept:
    indices = np.where(schaefer_map.get_fdata() == region)
    roi[indices] = 1

roi_img = nib.Nifti1Image(roi, schaefer_map.affine)
ns_roi = nilearn.image.resample_to_img(roi_img, neurosynth_dset.masker.mask_img, interpolation='nearest')

del roi_img

decoder = discrete.NeurosynthDecoder()
decoder.fit(neurosynth_dset)
ids_use = neurosynth_dset.get_studies_by_mask(ns_roi)
decode_df = decoder.transform(ids=ids_use) 
decode_df = decode_df.sort_values(by="probReverse", ascending=False)
decode_df.to_csv(f"../modeling/best_features/cog_terms/{n_features}_best_features_decode.csv")
sorted_terms = decode_df.index
sorted_terms = np.array([term.replace('terms_abstract_tfidf__','') for term in sorted_terms])

del decode_df
text =  " ".join(sorted_terms)
text = re.sub(r"\d+", "", text)
text = re.sub(r"LDA_", "", text)
text = text.replace("__", "")
text = text.replace("_", " ")
text = text.replace("  ", " ")

text_file = open(f"../modeling/best_features/cog_terms/{n_features}_best_features_cog_terms.txt", "w")
writer = text_file.write(text)
text_file.close()

print("Cog terms saved.")
