import numpy as np 
import nibabel as nib 
import pandas as pd
import pickle
import sys
from brainspace.gradient.alignment import ProcrustesAlignment

subj = sys.argv[1]
odir = sys.argv[2]
kernel = 4
        
clusterPath='/well/margulies/projects/data/hcpGrads'
anatNatPath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/Native'
anat32Path=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/fsaverage_LR32k'
MNIpath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/MNINonLinear/fsaverage_LR32k'

margulies_grads = np.load(f'{clusterPath}/margulies_grads_32k.npy') # 3 grads margulies 2016
alignment = ProcrustesAlignment()

gradses1=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
gradses2=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy') 

AllGrads = np.concatenate((gradses1[0], gradses2[0]))

allGradsAlignedObject = alignment.fit(AllGrads, margulies_grads) # aligning to margulies 2016
allGradsAligned = allGradsAlignedObject.aligned_
filename = odir +  "%s_aligned_grads.npy" % subj

np.save(arr = allGradsAligned, file = filename)