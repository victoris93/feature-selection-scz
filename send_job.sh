#!/bin/bash 
#SBATCH --job-name=gradicc
#SBATCH -o ./logs/gradicc-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array=1-912:1
#SBATCH --requeue



# I want the job to take 3 gradients of all 900 subjects, align them to Margulies 2016 and compite ICC for each vertex abd value of neighbours. Returns a huge csv with results, a long dataframe
module load Python/3.9.6-GCCcore-11.2.0 #do I need to do this with some packages/modules? install pingouin
source </victorias env>

SUBJECT_LIST=./SubjectsCompleteData.txt
ODIR = /well/margulies/projects/clinical_grads

echo "------------------------------------------------"
echo "Run on host: "hostname
echo "Operating system: "uname -s
echo "Username: "whoami
echo "Started at: "date
echo "------------------------------------------------"

python3 -u peaks2cortex.py $SUBJECT_LIST $ODIR # is that a correct way to call the script? Provided ODIR exists
