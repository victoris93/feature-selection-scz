import numpy as np 
import nibabel as nib 
import pandas as pd
import pickle
import sys
from brainspace.gradient.alignment import ProcrustesAlignment

### UTILS ###
### from: https://github.com/NeuroanatomyAndConnectivity/PkReliability/blob/main/hcp_class.py
### Author: https://github.com/neurabenn

def recort(X,fill,dims):
    out=np.zeros(dims)
    out[fill]=X
    return out

### HCP CLASS ###
class hcp_subj:
    def __init__(self,subj,kernel,pca=None,neighbours=None):
        
        self.subj=subj
        
        clusterPath='/well/margulies/projects/data/hcpGrads'
        anatNatPath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/Native'
        anat32Path=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/fsaverage_LR32k'
        MNIpath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/MNINonLinear/fsaverage_LR32k'
        
        self.info=np.load(f'{clusterPath}/{subj}/{subj}.cifti.info.npy',allow_pickle=True).item()
        
        self.dims=self.info['lnverts']
        self.Lfill=self.info['lIDX']
        self.Rfill=self.info['rIDX']
        self.pca=pca
        self.neighbours=neighbours
        
        self.Lsrf=f'{anat32Path}/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.LnatSrf=f'{anatNatPath}/{subj}.L.midthickness.native.surf.gii'
        self.Lcoords=nib.load(self.Lsrf).darrays[0].data
        self.Lfaces=nib.load(self.Lsrf).darrays[1].data
        
        self.Linflated=f'{anat32Path}/{subj}.L.inflated_MSMAll.32k_fs_LR.surf.gii'
        
        
        self.Rsrf=f'{anat32Path}/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.RnatSrf=f'{anatNatPath}/{subj}.R.midthickness.native.surf.gii'
        self.Rcoords=nib.load(self.Rsrf).darrays[0].data
        self.Rfaces=nib.load(self.Rsrf).darrays[1].data

        

        self.Laparc=f'{MNIpath}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'
        self.Lsulc=f'{MNIpath}/{subj}.L.sulc.32k_fs_LR.shape.gii'
        
        
        self.LV1=np.where(nib.load(self.Laparc).darrays[0].data==45)[0]
        self.LS1=np.where(nib.load(self.Laparc).darrays[0].data==46)[0]
        self.LA1=np.where(nib.load(self.Laparc).darrays[0].data==75)[0]
        
        
        self.Rinflated=f'{anat32Path}/{subj}.R.inflated_MSMAll.32k_fs_LR.surf.gii'

        
        self.Raparc=f'{MNIpath}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'
        self.Rsulc=f'{MNIpath}/{subj}.R.sulc.32k_fs_LR.shape.gii'
        
        self.RV1=np.where(nib.load(self.Raparc).darrays[0].data==45)[0]
        self.RS1=np.where(nib.load(self.Raparc).darrays[0].data==46)[0]
        self.RA1=np.where(nib.load(self.Raparc).darrays[0].data==75)[0]
        
#         self.LZverts=get_zoneVerts(LWS)
#         self.RZverts=get_zoneVerts(RWS)
    
#         self.LdistSens=np.load(f'{subj}/{subj}.L.dist32K.npy')
#         self.RdistSens=np.load(f'{subj}/{subj}.R.dist32K.npy')
        
        neighbours=self.neighbours
        
        if self.neighbours==None:
            pass
        else:
            self.Lneighbours=SpatialNeighbours(self.Lcoords,self.Lfaces)
            self.Rneighbours=SpatialNeighbours(self.Rcoords,self.Rfaces)
        
        
        if self.pca is None:
           #print('ussing diffusion maps')

            #### full gradient 
            self.grad=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.diffmaps.0{kernel}mm.npy')
            self.Lgrad=self.grad[0][0:len(self.Lfill)]
            self.Lgrad=recort(self.Lgrad,self.Lfill,self.dims)


            self.Rgrad=self.grad[0][len(self.Lfill):]
            self.Rgrad=recort(self.Rgrad,self.Rfill,self.dimssry)

            ###### session 1 
            ### subsessions
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
        
            ######## session 2 
 
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')
        
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            
        else:
#             print('using PCA maps')
            ######### load PCA grads
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses1.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses2.s0{kernel}mm.npy')   
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
    
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)

            ######### Aligned gradients w/ Procrustes
            self.AllGrads = np.concatenate((self.gradses1[0], self.gradses2[0]))

            margulies_grads = np.load(f'{clusterPath}/margulies_grads_32k.npy') # 3 grads margulies 2016
            alignment = ProcrustesAlignment()
            self.allGradsAlignedObject = alignment.fit(self.AllGrads, margulies_grads) # aligning to margulies 2016
            self.allGradsAligned = self.allGradsAlignedObject.aligned_

            self.Gradsses1Aligned = self.allGradsAligned[:int(len(self.allGradsAligned)/2)]
            self.Gradsses2Aligned = self.allGradsAligned[int(len(self.allGradsAligned)/2):]

            self.Lgradses1Aligned=self.Gradsses1Aligned[len(self.Lfill):]
            self.Lgradses1Aligned=recort(self.Lgradses1Aligned,self.Lfill,self.dims)

            self.Rgradses1Aligned=self.Gradsses1Aligned[len(self.Rfill):]
            self.Rgradses1Aligned=recort(self.Rgradses1Aligned,self.Rfill,self.dims)

            self.Lgradses2Aligned=self.Gradsses2Aligned[len(self.Lfill):]
            self.Lgradses2Aligned=recort(self.Lgradses2Aligned,self.Lfill,self.dims)

            self.Rgradses2Aligned=self.Gradsses2Aligned[len(self.Rfill):]
            self.Rgradses2Aligned=recort(self.Rgradses2Aligned,self.Rfill,self.dims)



    def print_subj(self):
        print(self.subj)

if __name__ == "__main__":
    subj = sys.argv[1]
    odir = sys.argv[2]
    hcp_subject_obj = hcp_subj(subj, 4)
    filname = odir + "/hcp_objects.pkl"
    filehandler = open(filname, 'wb') 
    pickle.dump(hcp_subject_obj, filehandler)
    filehandler.close()
