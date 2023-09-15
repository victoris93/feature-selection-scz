#!/bin/bash 
#SBATCH --job-name=Best1000VSgrad1
#SBATCH -o ./logs/Best1000VSgrad1-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=20

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the task id is $SLURM_ARRAY_JOB_ID
echo the job id is $SLURM_JOB_ID

python3 -u best1000_vs_grad1.py