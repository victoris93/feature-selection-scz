#!/bin/bash 
#SBATCH --job-name=NsynthAnalysis
#SBATCH -o ./logs/NsynthAnalysis-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-28:1 

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
ARG_LIST=./regions_nsynth.txt
echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

n_regions=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST)

python3 -u nsynth_analysis.py $n_regions