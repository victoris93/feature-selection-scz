#!/bin/bash 
#SBATCH --job-name=EigenvaluesSRPBS
#SBATCH -o ./logs/EigenvaluesSRPBS-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array 1-2:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 

site=$1
SUBJECT_LIST=/gpfs3/well/margulies/projects/data/SRPBS_1600/"${site}"/"SRPBS_1600_${site}"_subjects.txt
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo $sub $site
python -u get_eigenvalues.py $sub $site
