#!/bin/bash 
#SBATCH --job-name=ParcSchaefer1000
#SBATCH -o ./logs/ParcSchaefer1000-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-917:1

SUBJECT_LIST=./HCPSubjects.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID


sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

python -u parcellate_schaefer1000.py $sub