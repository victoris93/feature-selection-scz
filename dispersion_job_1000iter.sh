#!/bin/bash 
#SBATCH --job-name=1000iterComputeDispersion
#SBATCH -o ./logs/1000iterDispersion-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-1018:1

SUBJECT_LIST=./SubjectsCompleteData.txt

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)


python3 -u compute_dispersion_1000iter.py $SUBJECT
