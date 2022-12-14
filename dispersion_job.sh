#!/bin/bash
#$ -wd /well/margulies/projects/clinical_grads/Results
#$ -N DispersionDfs
#$ -j y
#$ -o /well/margulies/projects/clinical_grads/Results/GradDispersion
#$ -q short.qc
#$ -pe shmem 6
#$ -t 1:1018

mkdir -p /well/margulies/projects/clinical_grads/Results/GradDispersion
output_path=/well/margulies/projects/clinical_grads/Results/GradDispersion

SUBJECT_LIST=./SubjectsCompleteData.txt

echo Executing task ${ARRAY_TASK_ID} of job ${ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $ARRAY_JOB_ID
FILENAME=$(sed -n "${ARRAY_TASK_ID}p" $SUBJECT_LIST)

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

echo python3 -u dispersion_df.py $FILENAME $output_path
python3 -u dispersion_df.py $FILENAME $output_path