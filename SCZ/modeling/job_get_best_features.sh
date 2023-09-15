#!/bin/bash 
#SBATCH --job-name=GetBestFeatures
#SBATCH -o ./logs/GetBestFeatures-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-1000:1 

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
ARG_LIST=./n_features.txt
echo the job id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

n_features=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST)

python3 -u get_best_features.py $n_features