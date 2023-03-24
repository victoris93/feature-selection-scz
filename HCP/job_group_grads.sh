#!/bin/bash 
#SBATCH --job-name=GradsHCP
#SBATCH -o ./logs/GradsHCP-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER}

python -u group_grads_hcp.py
