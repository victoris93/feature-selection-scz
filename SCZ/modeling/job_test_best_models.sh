#!/bin/bash 
#SBATCH --job-name=TestBestModel
#SBATCH -o ./logs/TestBestModel-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=15
#SBATCH --array 1-100:1 

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
ARG_LIST=./n_features.txt
echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

n_features=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST)

python3 -u test_best_models.py $1 $n_features