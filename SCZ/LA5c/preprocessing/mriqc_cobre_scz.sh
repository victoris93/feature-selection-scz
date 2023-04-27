#!/bin/bash 
#SBATCH --job-name=MRIQC
#SBATCH -o ./logs/MRIQC-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=3
#SBATCH --array 1-73:1

SUBJECT_LIST=/gpfs3/well/margulies/projects/data/COBRE/SCZ/COBRE_SCZ_subjects.txt

SUBJECT_LIST=/gpfs3/well/margulies/projects/data/COBRE/SCZ/COBRE_SCZ_subjects.txt
DATA=/gpfs3/well/margulies/projects/data/COBRE/SCZ
OUTPUT=/gpfs3/well/margulies/projects/data/COBRE/SCZ/MRIQC


echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

singularity run --cleanenv --bind --bind /well/margulies/projects/data/ $DATA:/data --bind $OUTPUT:/out \
<mriqc_latest.sif> \
/data /out participant \
--participant_label $sub
