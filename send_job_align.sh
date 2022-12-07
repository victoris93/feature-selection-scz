#!/bin/bash 

#SBATCH --job-name=GradAlign
#SBATCH -o ./logs/GradAlign-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --array 1-5:1
SUBJECT_LIST=./SubjectsCompleteData.txt

# I want the job to take 3 gradients of all 900 subjects, align them to Margulies 2016 and compite ICC for each vertex abd value of neighbours. Returns a huge csv with results, a long dataframe
module load Python/3.9.6-GCCcore-11.2.0 #do I need to do this with some packages/modules? install pingouin
source /well/margulies/users/mnk884/Victoria/ClinicalGradients-skl/bin/activate

mkdir -p /well/margulies/projects/clinical_grads/Results
output_path=/well/margulies/projects/clinical_grads/Results
echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID
FILENAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
#echo echo $SLURM_ARRAY_JOB_ID
#echo "Processing subject $FILENAME"

echo python3 -u alignment.py $FILENAME  $output_path
python3 -u alignment.py $FILENAME  $output_path
