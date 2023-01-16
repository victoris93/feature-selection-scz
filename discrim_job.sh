#!/bin/bash 
#SBATCH --job-name=ComputeDiscrim
#SBATCH -o ./logs/ComputeDiscrim-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

echo the job id is $SLURM_ARRAY_JOB_ID

python3 -u compute_discriminability.py