#!/bin/bash 
#SBATCH --job-name=MRIQC
#SBATCH -o ./logs/MRIQC-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=3
#SBATCH --array 1-73:1

SUBJECT_LIST=/gpfs3/well/margulies/projects/data/COBRE/SCZ/COBRE_SCZ_subjects.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/mriqc-env/bin/activate

SUBJECT_LIST=/gpfs3/well/margulies/projects/data/COBRE/SCZ/COBRE_SCZ_subjects.txt
DATA=/gpfs3/well/margulies/projects/data/COBRE/SCZ
OUTPUT=/gpfs3/well/margulies/projects/data/COBRE/SCZ/MRIQC
export TEMPLATEFLOW_HOME=/gpfs3/well/margulies/users/cpy397/templateflow


echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

mriqc $DATA $OUTPUT participant --participant-label $sub -vvv
