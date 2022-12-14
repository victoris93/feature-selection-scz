#!/bin/bash
#$ -wd /gpfs3/well/margulies/users/anw410/Vic
#$ -N DespReliabilityICC
#$ -j y
#$ -o /gpfs3/well/margulies/users/anw410/Vic/log
#$ -q short.qc
#$ -pe shmem 6
#$ -t 1:9

NEIGHBOURS_LIST=/gpfs3/well/margulies/users/anw410/Vic/n_neighbours.txt

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

FILENAME=$(sed -n "${SGE_TASK_ID}p" $NEIGHBOURS_LIST)

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

python3 -u vertex_ICC_32k.py $FILENAME
