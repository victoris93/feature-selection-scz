#!/bin/bash
#$ -wd /gpfs3/well/margulies/users/anw410/Vic
#$ -N DispVertexICC
#$ -j y
#$ -o /gpfs3/well/margulies/users/anw410/Vic/log
#$ -q short.qc
#$ -pe shmem 4
#$ -t 1-60

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

python -u vertex_ICC_32k.py ${SGE_TASK_ID} $1
