#!/bin/bash 
#SBATCH --job-name=ConfModels
#SBATCH -o ./logs/ConfModels-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the job id is $SLURM_JOB_ID

python3 -u confounds_model_cv_test.py