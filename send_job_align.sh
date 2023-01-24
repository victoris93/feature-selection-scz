#!/bin/bash 
#SBATCH --job-name=GradAlign
#SBATCH -o ./logs/GradAlign-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --array 1-1018:1

SUBJECT_LIST=./SubjectsCompleteData.txt

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

output_path=/gpfs3/well/margulies/users/cpy397/AlignedGradsPCA/1000_iter
echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
FILENAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

echo python3 -u alignment.py $FILENAME  $output_path
python3 -u alignment.py $FILENAME  $output_path
