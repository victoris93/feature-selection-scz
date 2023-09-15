#!/bin/bash 
#SBATCH --job-name=BestConn
#SBATCH -o ./logs/BestConn-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=10

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

python3 -u fit_best1000_conn.py