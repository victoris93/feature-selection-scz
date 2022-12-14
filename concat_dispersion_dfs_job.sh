#!/bin/bash
#$ -wd /gpfs3/well/margulies/users/anw410/Vic
#$ -N ConcatDispersionDfs
#$ -j y
#$ -o /gpfs3/well/margulies/users/anw410/Vic/log
#$ -q short.qc
#$ -pe shmem 6
#$ -t 1:1018

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

N_Neighbours=(50 100 200 400 800 1600 3200 6400 12800)

SUBJECT_LIST=./SubjectsCompleteData.txt

for neighbours in ${N_Neighbours[@]}; do

    FILENAME=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)
    N_Neighbours=neighbours
    python3 -u concatenate_dispersion_dfs.py $FILENAME $neighbours
