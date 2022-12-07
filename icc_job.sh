#!/bin/bash 

#SBATCH --job-name=ICC
#SBATCH -o ./logs/ICC-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --requeue

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

mkdir -p /well/margulies/projects/clinical_grads/Results/vertex_icc

output_path=/well/margulies/projects/clinical_grads/Results/vertex_icc
ARRAY=/well/margulies/projects/clinical_grads/Results/aligned_grads.npy

python3 -u vertex_ICC_32k.py $ARRAY $output_path
