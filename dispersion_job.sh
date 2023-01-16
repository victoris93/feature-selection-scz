#!/bin/bash 
#SBATCH --job-name=ComputeDispersion
#SBATCH -o ./logs/Dispersion-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=5
#SBATCH --array 1-500:1

SUBJECT_LIST=./SubjectsCompleteData.txt

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

output_path=/gpfs3/well/margulies/users/cpy397/DispersionResults
echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)


python3 -u compute_dispersion.py $SUBJECT
