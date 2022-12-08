import numpy as np 
import os
import nibabel as nib 
import pandas as pd
import pickle
import sys
from brainspace.gradient.alignment import ProcrustesAlignment

subj = sys.argv[1]
#subj = "100206"
odir = sys.argv[2]
kernel = 4
        
clusterPath='/well/margulies/projects/data/hcpGrads'

margulies_grads = np.load(f'{clusterPath}/margulies_grads_32k.npy') # 3 grads margulies 2016
#margulies_grads = np.load('margulies_grads_32K.npy') # 3 grads margulies 2016

alignment = ProcrustesAlignment(verbose=True)

gradses1=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
gradses2=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')

#gradses1=np.load(f'HCP/Gradients/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
#gradses2=np.load(f'HCP/Gradients/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')

AllGrads = np.stack((gradses1, gradses2)).reshape(2, 59412, 3)
#print(margulies_grads.shape)
 
allGradsAlignedObject = alignment.fit(AllGrads, margulies_grads.T) # aligning to margulies 2016
allGradsAligned = allGradsAlignedObject.aligned_

np.save(arr = allGradsAligned, file =f"{clusterPath}/{subj}.GradsAligned2Margulies2016.npy")
#np.save(arr = allGradsAligned, file ="GradsAligned2Margulies2016.npy")

