#!/bin/bash 
#SBATCH --job-name=GradAnalysis
#SBATCH -o ./logs/GradAnalysis-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=10

module load Python/3.9.6-GCCcore-11.2.0
source /users/margulies/cpy397/env/ClinicalGrads/bin/activate
### source /well/margulies/users/mnk884/python/corrmats-skylake/bin/activate

output_path=/gpfs3/well/margulies/users/cpy397/DispersionResults
echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

python -u grads_1subj.py
