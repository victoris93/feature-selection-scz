#!/bin/bash 
#SBATCH --job-name=ComputeDiscrim
#SBATCH -o ./logs/ComputeDiscrim-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array 1-9:1

NEIGHBOURS=./neighbours.txt

module load Python/3.9.6-GCCcore-11.2.0
source /gpfs3/users/margulies/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

N_NEIGHBOURS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $NEIGHBOURS)
python3 -u compute_discriminability.py $N_NEIGHBOURS