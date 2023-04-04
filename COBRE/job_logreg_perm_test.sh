#!/bin/bash 
#SBATCH --job-name=PermTestLogReg
#SBATCH -o ./logs/PermTestLogReg-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array 1-1000:1

REGION_LIST=./schaefer1000_regions.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

REGION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $REGION_LIST)

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
echo REGION is $REGION

python -u log_reg_perm_test.py $REGION