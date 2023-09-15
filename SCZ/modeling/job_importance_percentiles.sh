#!/bin/bash 
#SBATCH --job-name=ImpPercent
#SBATCH -o ./logs/ImpPercent-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=15

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

python3 -u importance_percentiles.py $1