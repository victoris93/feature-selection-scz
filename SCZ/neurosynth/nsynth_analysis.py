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

region = int(sys.argv[1]) + 1

neurosynth_dset = Dataset.load("neurosynth_dataset.pkl.gz")

schaefer_map = nilearn.datasets.fetch_atlas_schaefer_2018(1000, yeo_networks = 7)
schaefer_map = nb.load(schaefer_map['maps'])

roi = np.zeros_like(schaefer_map.get_fdata())
print("Running Nsynth for region: ", region)

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
decode_df["Region"] = region - 1
decode_df.to_csv(f"../modeling/best_features/cog_terms/region{region - 1}_nsynth_decode.csv")

print(f"Cog terms saved for region {region}.")
