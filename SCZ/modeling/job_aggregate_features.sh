#!/bin/bash 
#SBATCH --job-name=AggregateFeatures
#SBATCH -o ./logs/AggregateFeatures-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=24
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the job id is $SLURM_JOB_ID
python3 -u aggregate_features.py