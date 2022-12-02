#!/usr/bin/env python
import os
import sys
import pathlib
import argparse
import nibabel as nib
import numpy as np
import brainspace as bs
import subprocess as sp 
import cython
from nilearn import signal
from scipy import spatial
import gdist as gd
from brainspace.gradient import GradientMaps
from sklearn.metrics import pairwise_distances
from surfdist.utils import find_node_match
np.set_printoptions(suppress=True)

####### define the functions we'll be using 

def save_gifti(data,out):
	"""Save gifti file providing a numpy array and an output file path"""
	gi = nib.gifti.GiftiImage()
	da = nib.gifti.GiftiDataArray(np.float32(data), intent=0)
	gi.add_gifti_data_array(da)
	nib.save(gi,f'{out}.func.gii')


def post_smooth(func):
	"""Zscore normalize, bandpass filter, and remove first 10 volumes"""
	cifti=nib.load(func)
	#### clean the time series 
	cln=signal.clean(cifti.get_fdata(),detrend=True,standardize='zscore',filter='butterworth',low_pass=0.08,high_pass=0.008)
	return cln[10:]

def wb_smoothCleanTs(subject,func_dat,kernel,leftSrf,rightSrf):
	"""" Smoooth, Normalize and Bandpass Filter data """
# 	inter=func_dat.split('dtseries.nii')[0]+f'{kernel}mm.dtseries.nii' #### named inter because file will be deleted
	tempStorage='/well/margulies/users/mnk884/PkReliability/tempFiles'
	
	inter=f'{tempStorage}/{subject}.0{kernel}mm.dtseries.nii' #### implementation for using hcp data on the cluster.
	print(f' intermediate smoothed time series is {inter}')
	#### one of the few times we'll be writing data
	print(f'the intermediate file out is {inter}')
	cmd=f'wb_command -cifti-smoothing {func_dat} {kernel} {kernel} COLUMN {inter} -left-surface {leftSrf} -right-surface {rightSrf}'
	sp.run(cmd,shell=True)
	clnTs=post_smooth(inter)
	sp.run(f'rm {inter}',shell=True) 
	return clnTs

def get_corticalVertices(data):
	""" Get indices of Cortex Data from cifti file """
	cifti=nib.load(data)
	structMap=cifti.header.get_index_map(1)
	brainModels=list(structMap.brain_models)
	LCrtBM=brainModels[0]
	Lcrt_vrts=np.array(LCrtBM.vertex_indices)
	LnumVerts=LCrtBM.surface_number_of_vertices
	
	RCrtBM=brainModels[1]
	Rcrt_vrts=np.array(RCrtBM.vertex_indices)
	RnumVerts=RCrtBM.surface_number_of_vertices
	
	return {'lIDX':Lcrt_vrts,'lnverts':LnumVerts,'rIDX':Rcrt_vrts,'rnverts':RnumVerts}

def concat_sessions(DataList):
	return np.vstack([DataList[0],DataList[1]]).T, np.vstack([DataList[2],DataList[3]]).T

def pick_cortex(data,label):
	##### standard slices for HCP data and the 32k surface the time series are mapped on to 
	lcort=slice(0,29696)
	rcort=slice(29696, 59412)
	cortAll=slice(0,59412)
	###### slice time series based on hemisphere choice
	if label=='left':
		print('Using Left Cortex Only')
		data=data[lcort]
	elif label=='right':
		print('Using Right Cortex Only')
		data=data[rcort]
	else:
		print('Using whole cortex')
		data=data[cortAll]
	return data
		
####### use only if running locally. 
def calcFC(data):
	return np.corrcoef(data)


def calcFC_chunks(data):
	bigdata=data
	bigdata -= np.mean(bigdata, axis=1)[:,None]
	bigdata /= np.sqrt(np.sum(bigdata*bigdata, axis=1))[:,None]
	SPLITROWS = 1000
	numrows = bigdata.shape[0]
	res = np.memmap(f'{odir}/tmp.dat', 'float64', mode='w+', shape=(numrows, numrows))
	
	for r in range(0, numrows, SPLITROWS):
		for c in range(0, numrows, SPLITROWS):
			r1 = r + SPLITROWS
			c1 = c + SPLITROWS
			chunk1 = bigdata[r:r1]
			chunk2 = bigdata[c:c1]
			res[r:r1, c:c1] = np.dot(chunk1, chunk2.T)
	return res

def threshMat(conn,lim):
	perc = np.array([np.percentile(x, lim) for x in conn])
	# Threshold each row of the matrix by setting values below X percentile to 0
	for i in range(conn.shape[0]):
		conn[i, conn[i,:] < perc[i]] = 0   
	return conn

def pcaGrad(data):
	pca = GradientMaps(n_components=1, random_state=0,approach='pca')
	pca.fit(data)
	return pca.gradients_[:].squeeze()

def DiffEmbed(data,ngrads):
	""" FC matrix and number of gradients"""
	####input is threshold FC matrix
# 	aff = 1 - pairwise_distances(data, metric = 'cosine')
	dm = GradientMaps(n_components=ngrads, random_state=42,approach='dm',kernel='cosine')
	dm.fit(data)
	return dm.gradients_


def save_grads(pcaMap,deMap,idxMap,session,hemi=''):
    if hemi=='left':
        lpc=np.zeros(idxMap[0]['lnverts'])
        lpc[idxMap[0]['lIDX']]=pcaMap
        save_gifti(lpc,f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes{session}')
        
        lde=np.zeros(idxMap[0]['lnverts'])
        lde[idxMap[0]['lIDX']]=deMap
        save_gifti(lde,f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes{session}')
        return lpc,lde
    elif hemi=='right':
        rpc=np.zeros(idxMap[0]['rnverts'])
        rpc[idxMap[0]['rIDX']]=pcaMap
        save_gifti(rpc,f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes{session}')
        
        rde=np.zeros(idxMap[0]['rnverts'])
        rde[idxMap[0]['rIDX']]=deMap
        save_gifti(rde,f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes{session}')

        
        return rpc,rde
    else: 
        lpcaMap=pcaMap[lcort]
        ldeMap=deMap[lcort]
        
        lpc=np.zeros(idxMap[0]['lnverts'])
        lpc[idxMap[0]['lIDX']]=lpcaMap
        save_gifti(lpc,f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes{session}')
        
        lde=np.zeros(idxMap[0]['lnverts'])
        lde[idxMap[0]['lIDX']]=ldeMap
        save_gifti(lde,f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes{session}')
        
        rpcaMap=pcaMap[rcort]
        rdeMap=deMap[rcort]
        rpc=np.zeros(idxMap[0]['rnverts'])
        rpc[idxMap[0]['rIDX']]=rpcaMap
        save_gifti(rpc,f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes{session}')
        
        rde=np.zeros(idxMap[0]['rnverts'])
        rde[idxMap[0]['rIDX']]=rdeMap
        
        save_gifti(rde,f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes{session}')
        return lpc,lde,rpc,rde

def load_grads(dir,hemi):
	if hemi == 'left':
		PC1=f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes01.func.gii'
		PC2=f'{odir}/{subj}.L.PCA.G1.{kernel}mmTsSes02.func.gii'

		DE1=f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes01.func.gii'
		DE2=f'{odir}/{subj}.L.DE.G1.{kernel}mmTsSes02.func.gii'
		return PC1,PC2,DE1,DE2
	elif hemi =='right':
		print('running right hemi')
		PC1=f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes01.func.gii'
		PC2=f'{odir}/{subj}.R.PCA.G1.{kernel}mmTsSes02.func.gii'

		DE1=f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes01.func.gii'
		DE2=f'{odir}/{subj}.R.DE.G1.{kernel}mmTsSes02.func.gii'
		return PC1,PC2,DE1,DE2


def smooth_grad(grad,hemi):
	##### first set up a label to only smooth over cortical values 
	mask=nib.load(grad).agg_data()
	mask[np.where(mask!=0)[0]]=1
	save_gifti(mask,f'{odir}/cortexmask')

	base=grad.split('.func.gii')[0]
	
	gradSet=[grad]

	for krnl in [4,8,12,16]:
		if hemi =='left':
			gradSet.append(f'{base}_{krnl}mm.func.gii')
			cmd=f'wb_command -metric-smoothing {Lsrf32} {grad} {krnl} {base}_{krnl}mm.func.gii -roi {odir}/cortexmask.func.gii'
			sp.run(cmd,shell=True)
		elif hemi == 'right':
			gradSet.append(f'{base}_{krnl}mm.func.gii')
			cmd=f'wb_command -metric-smoothing {Rsrf32} {grad} {krnl} {base}_{krnl}mm.func.gii -roi {odir}/cortexmask.func.gii'
			sp.run(cmd,shell=True)
	return gradSet

def gradientOrientation(grad,hemi):
	"""Determine the orientation of the gradients, and also return whether valid for continued study or not"""
	grad=nib.load(grad).agg_data()
	if hemi=='left':
		print('getting gradient orientation from Left hemisphere')
		labels=nib.load(Laparc).agg_data()
		calc=np.where(labels==45)[0]
		ctr=np.where(labels==46)[0]
		if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
			print('Canonical Orientation DMN at apex')
			return grad,True
		elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
			print(f'REMOVE {subj} FROM STUDY')
			return grad,False
		elif np.sum(grad[calc])>0 and np.sum(grad[ctr])<0:
			print(f'REMOVE {subj} FROM STUDY')
			return grad,False
		else:
			print('flipping gradient orientation for peak detection')
			return grad *-1,True
		
	elif hemi=='right':
		print('getting gradient orientation from Right hemisphere')
		labels=nib.load(Raparc).agg_data()
		calc=np.where(labels==45)[0]
		ctr=np.where(labels==46)[0]
		if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
			print('Canonical Orientation DMN at apex')
			return grad,True
		elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
			print(f'REMOVE {subj} FROM STUDY')
			return grad,False
		elif np.sum(grad[calc])>0 and np.sum(grad[ctr])<0:
			print(f'REMOVE {subj} FROM STUDY')
			return grad,False
		else:
			print('flipping gradient orientation before peak detection')
			return grad *-1,True


def get_peaks(grad,zoneParc):
	labels=zoneParc
	Lpar=np.where(labels==2)[0]
	Ltmp=np.where(labels==5)[0]
	Mpar=np.where(labels==7)[0]
	pks=[]
	for i in [Lpar,Ltmp,Mpar]:
		pks.append(i[np.argmax(grad[i])])
	return pks

def lookupHires(hemi):

	if hemi =='left':
		surf1=nib.load(Lsrf32).darrays[0].data
		surf2=nib.load(LsrfNative).darrays[0].data

		srf_mathced=find_node_match(surf1,surf2)[0]
	
		return srf_mathced
	elif hemi =='right':
		surf1=nib.load(Rsrf32).darrays[0].data
		surf2=nib.load(RsrfNative).darrays[0].data

		srf_mathced=find_node_match(surf1,surf2)[0]
	
		return srf_mathced


def dist_btw_pks(set1,set2,surf):
	verts=nib.load(surf).darrays[0].data.astype('float64')
	faces=nib.load(surf).darrays[1].data.astype('int32')
	dist_out=[]
	for i,j in zip(set1,set2):
		i=np.asarray([i]).astype('int32')
		j=np.asarray([j]).astype('int32')
		print(i,j)
		dist_out.append(gd.compute_gdist(verts,faces,i,j))
	return np.asarray(dist_out).squeeze()

def getXsessionPkDist(ses_grads,hemi,Nat_32):
	if hemi == 'left':
		#### insert function for smoothing here? 

		ws=LWS
		### get peaks 32K space 
		pkset1=get_peaks(ses_grads[0],ws)
		pkset2=get_peaks(ses_grads[1],ws)

		pkset1Nat=[Nat_32[z] for z in pkset1]
		pkset2Nat=[Nat_32[z] for z in pkset2]

		print(pkset1,pkset1Nat)
		print(pkset2,pkset2Nat)
		return np.asarray(dist_btw_pks(pkset1Nat,pkset2Nat,LsrfNative))

	elif hemi =='right':

		#### insert function for smoothing here?

		ws=RWS
		pkset1=get_peaks(ses_grads[0],ws)
		pkset2=get_peaks(ses_grads[1],ws)

		pkset1Nat=[Nat_32[z] for z in pkset1]
		pkset2Nat=[Nat_32[z] for z in pkset2]

		print(pkset1,pkset1Nat)
		print(pkset2,pkset2Nat)

		# return np.asarray([pkset1,pkset2,pkset1Nat,pkset2Nat,dist_btw_pks(pkset1Nat,pkset2Nat,RsrfNative)])
		return np.asarray(dist_btw_pks(pkset1Nat,pkset2Nat,RsrfNative))


def post_embed(hemi):
	if hemi != 'left' and  hemi !='right':
		print('doing on both hemispheres')

		for hemi in ['left','right']:
		### first step get in the raw files and check orientation 
			pc1,pc2,de1,de2=load_grads(subj,hemi)

			print(pc1,pc2,de1,de2)
			pc=[pc1,pc2]
			de=[de1,de2]	

			pca_valid=[]
			pca_flip=[]
			for comp in range(len(pc)):
				print(pc[comp])
				pc[comp],val=gradientOrientation(pc[comp],hemi)
				pca_valid.append(val)
			de_valid=[]
			de_flip=[]
			for comp in range(len(de)):
				print(de[comp])
				de[comp],val=gradientOrientation(de[comp],hemi)
				de_valid.append(val)

			#### write for. diffusion embedding as that one works, then add in for PCA 
		
		
			if False in pca_valid:
				print(' PCA has found at least one session\'s principal gradient separates sensory modalities. \n \
				Non-canonical orientation, Peak detection will not be run.\n')
			else: 
				ses1Set=smooth_grad(de1,hemi)
				ses2Set=smooth_grad(de2,hemi)

				Nat32=lookupHires(hemi)

				out=[]
				for i,j in zip(ses1Set,ses2Set):
					base=i.split('.func.gii')[0]
					subset=[gradientOrientation(i,hemi)[0],gradientOrientation(j,hemi)[0]]
					out.append(getXsessionPkDist(subset,hemi,Nat32))
				np.save(f'{odir}/{subj}.{hemi}.DE.peaks+dists',np.asarray(out))


		if False in de_valid:
			print(' Diffusion Mapping has found at least one session\'s principal gradient separates sensory modalities. \n \
			Non-canonical orientation, Peak detection will not be run.\n')
		else: 
			ses1Set=smooth_grad(de1,hemi)
			ses2Set=smooth_grad(de2,hemi)

			Nat32=lookupHires(hemi)

			out=[]
			for i,j in zip(ses1Set,ses2Set):
				base=i.split('.func.gii')[0]
				subset=[gradientOrientation(i,hemi)[0],gradientOrientation(j,hemi)[0]]
				out.append(getXsessionPkDist(subset,hemi,Nat32))
			np.save(f'{odir}/{subj}.{hemi}.DE.peaks+dists',np.asarray(out))




	else:
		### first step get in the raw files and check orientation 
		pc1,pc2,de1,de2=load_grads(subj,hemi)

		print(pc1,pc2,de1,de2)
		pc=[pc1,pc2]
		de=[de1,de2]	

		pca_valid=[]
		pca_flip=[]
		for comp in range(len(pc)):
			print(pc[comp])
			pc[comp],val=gradientOrientation(pc[comp],hemi)
			pca_valid.append(val)
		de_valid=[]
		de_flip=[]
		for comp in range(len(de)):
			print(de[comp])
			de[comp],val=gradientOrientation(de[comp],hemi)
			de_valid.append(val)

			#### write for. diffusion embedding as that one works, then add in for PCA 
		
		
		if False in pca_valid:
			print(' PCA has found at least one session\'s principal gradient separates sensory modalities. \n \
			Non-canonical orientation, Peak detection will not be run.\n')
		else: 
			ses1Set=smooth_grad(de1,hemi)
			ses2Set=smooth_grad(de2,hemi)

			Nat32=lookupHires(hemi)

			out=[]
			for i,j in zip(ses1Set,ses2Set):
				base=i.split('.func.gii')[0]
				subset=[gradientOrientation(i,hemi)[0],gradientOrientation(j,hemi)[0]]
				out.append(getXsessionPkDist(subset,hemi,Nat32))
			np.save(f'{odir}/{subj}.{hemi}.DE.peaks+dists',np.asarray(out))


		if False in de_valid:
			print(' Diffusion Mapping has found at least one session\'s principal gradient separates sensory modalities. \n \
			Non-canonical orientation, Peak detection will not be run.\n')
		else: 
			ses1Set=smooth_grad(de1,hemi)
			ses2Set=smooth_grad(de2,hemi)

			Nat32=lookupHires(hemi)

			out=[]
			for i,j in zip(ses1Set,ses2Set):
				base=i.split('.func.gii')[0]
				subset=[gradientOrientation(i,hemi)[0],gradientOrientation(j,hemi)[0]]
				out.append(getXsessionPkDist(subset,hemi,Nat32))
			np.save(f'{odir}/{subj}.{hemi}.DE.peaks+dists',np.asarray(out))