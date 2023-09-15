#!/bin/bash 
#SBATCH --job-name=Dispersion
#SBATCH -o ./logs/Dispersion-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array 1-100001:1

ARG_LIST=./disp_params.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

N_GRADS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $1}')
N_NEIGHBOURS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $2}')

echo Computing dispersion from $1 $N_GRADS gradient and $N_NEIGHBOURS closest neighbours
python3 -u compute_dispersion.py $N_GRADS $N_NEIGHBOURS $1 $2