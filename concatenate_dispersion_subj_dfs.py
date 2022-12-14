import pandas as pd
import sys
import os

# do that for each n_neighbours

subject = sys.argv[1]
n_neighbours = sys.argv[2]
odir=r'/gpfs3/well/margulies/users/anw410/Vic/Results' 

output_file = f"{odir}/GradDispersionDf_{n_neighbours}Neighbours.csv" 

dfExists = os.path.exists(output_file)
SubjDispersionDf = pd.read_csv(os.path.join(odir, 'dispersion_'+str(n_neighbours)+'_neighbours_'+str(subject)+'_df.csv'))

#check whether the previous version of the dataframe exists

if not dfExists: #if it doesn't, save the first subject as such
    SubjDispersionDf.to_csv(output_file) 
else: # if it does, concatenate the rest of the subjects to it and overwrite
    DispersionDf = pd.read_csv(output_file)
    DispersionDf = pd.concat((DispersionDf, SubjDispersionDf))
    DispersionDf.to_csv(output_file)

