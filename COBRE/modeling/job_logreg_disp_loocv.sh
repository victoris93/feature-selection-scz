#!/bin/bash 
#SBATCH --job-name=LogRegDisp
#SBATCH -o ./logs/LogRegDisp-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array 1-1000:1

ARG_LIST=./args_panss_linreg.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

N_GRADS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $1}')
N_NEIGHBOURS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $ARG_LIST | awk '{print $2}')

echo Fitting logistic regression on dispersion \& SCZ diagnosis computed from $N_GRADS th gradient and $N_NEIGHBOURS closest neighbours

python3 -u logreg_loocv_disp.py $N_GRADS $N_NEIGHBOURS