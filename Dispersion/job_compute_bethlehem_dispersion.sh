#!/bin/bash 
#SBATCH --job-name=BthlhmDisp
#SBATCH -o ./logs/BthlhmDisp-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array 1-1000:1


module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

DATA_PATH=$1
SUBJECT_LIST=$2
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

echo Computing within- and between-network dispersion for subject $sub from dataset $DATA_PATH
python3 -u bethlehem_dispersion.py $sub $DATA_PATH