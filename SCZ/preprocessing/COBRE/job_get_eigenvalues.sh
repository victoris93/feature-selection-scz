#!/bin/bash 
#SBATCH --job-name=EigenvaluesCOBRE
#SBATCH -o ./logs/EigenvaluesCOBRE-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array 1-91:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
### source /well/margulies/users/mnk884/python/corrmats-skylake/bin/activate

echo Executing job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

group=$1
SUBJECT_LIST=/gpfs3/well/margulies/projects/data/COBRE/"${group}"/COBRE_"${group}"_subjects.txt
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

python -u get_eigenvalues.py $sub $group
