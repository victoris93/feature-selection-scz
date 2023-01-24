import numpy as np 
import os
import nibabel as nib 
import pandas as pd
import pickle
import sys
from brainspace.gradient.alignment import ProcrustesAlignment

subj = sys.argv[1]
output_path = sys.argv[2]
kernel = 4
        
clusterPath='/well/margulies/projects/data/hcpGrads'

margulies_grads = np.load(f'{clusterPath}/margulies_grads_32k.npy') # 3 grads margulies 2016
#margulies_grads = np.load('margulies_grads_32K.npy') # 3 grads margulies 2016

alignment = ProcrustesAlignment(n_iter = 1000, tol = 0, verbose=True)

gradses1=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses1.s0{kernel}mm.npy').T
gradses2=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses2.s0{kernel}mm.npy').T
 
allGradsAlignedObject = alignment.fit([gradses1,gradses2], margulies_grads.T) # aligning to margulies 2016
allGradsAligned = allGradsAlignedObject.aligned_

np.save(arr = allGradsAligned, file =f"{output_path}/{subj}.PCAGradsAligned2Margulies2016_1000iter.npy")