from icc_utils import gradDispersion
import os,sys
import numpy as np 


clusterPath = '/gpfs3/well/margulies/users/cpy397/AlignedGradsPCA/1000_iter'
# icc_utils should be here: '/well/margulies/projects/clinical_grads/Results'

subject = sys.argv[1]
odir = r'/gpfs3/well/margulies/users/cpy397/DispersionResults/1000_iter'

#print(odir)
neighbours = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
grads_aligned = np.load(f"{clusterPath}/{subject}.PCAGradsAligned2Margulies2016_1000iter.npy")
print(grads_aligned.shape)

for n_neighbours in neighbours:
    if not os.path.exists(odir + "/PCAGradDispersion_%s_%s_neighbours_1000iter.npy" % (subject, n_neighbours)):
        print("Computing dispersion for subject %s" % subject)
        dispersion = gradDispersion(grads_aligned[0], grads_aligned[1], n_neighbours = n_neighbours)
        np.save(arr = dispersion, file = odir + "/PCAGradDispersion_%s_%s_neighbours_1000iter.npy" % (subject, n_neighbours))
        print("Saved PCAGradDispersion_%s_%s_neighbours_1000iter.npy in %s" % (subject, n_neighbours, odir))


#for n_neighbours in neighbours:
#    print("Computing dispersion for subject %s" % subject)
#    dispersion = gradDispersion(grads_aligned[0], grads_aligned[1], n_neighbours = n_neighbours)
#   np.save(arr = dispersion, file = odir + "/PCAGradDispersion_%s_%s_neighbours_1000iter.npy" % (subject, n_neighbours))
#    print("Saved PCAGradDispersion_%s_%s_neighbours_1000iter.npy in %s" % (subject, n_neighbours, odir))