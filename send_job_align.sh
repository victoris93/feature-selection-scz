#!/bin/bash 

#SBATCH --job-name=GradAlign
#SBATCH -p short
#$ -t 1-527:1

# I want the job to take 3 gradients of all 900 subjects, align them to Margulies 2016 and compite ICC for each vertex abd value of neighbours. Returns a huge csv with results, a long dataframe
module load Python/3.9.6-GCCcore-11.2.0 #do I need to do this with some packages/modules? install pingouin
source </victorias env>

echo "------------------------------------------------"
echo "Run on host: "hostname
echo "Operating system: "uname -s
echo "Username: "whoami
echo "Started at: "date
echo "------------------------------------------------"

output_path = /well/margulies/projects/clinical_grads/Results

for subject in 'cat SubjectsCompleteData.txt';do
    python3 -u hcp_class_alignment.py $subject $output_path
done
