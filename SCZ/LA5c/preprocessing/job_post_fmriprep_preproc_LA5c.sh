#!/bin/bash 
#SBATCH --job-name=LA5cPostFmriprep
#SBATCH -o ./logs/LA5cPostFmriprep-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array 1-272:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

SUBJECT_LIST=/gpfs3/well/margulies/projects/data/LA5c/la5c_subjects.txt
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

python -u post_fmriprep_preproc_LA5c.py $sub
