#!/bin/bash 
#SBATCH --job-name=RandFeatTest
#SBATCH -o ./logs/RandFeatTest-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-954:1 

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
ARG_LIST=./random_test_args.txt
echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

model=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $1}')
n_features=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $2}')

python3 -u random_feature_test.py $model $n_features