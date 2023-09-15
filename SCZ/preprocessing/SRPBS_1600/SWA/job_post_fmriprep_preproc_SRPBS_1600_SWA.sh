#!/bin/bash 
#SBATCH --job-name=SWAPostFmriprep
#SBATCH -o ./logs/SWAPostFmriprep-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array 1-235:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

export MRIS=/gpfs3/well/margulies/projects/data/SRPBS_1600/SWA
SUBJECT_LIST=/gpfs3/well/margulies/projects/data/SRPBS_1600/SWA/SRPBS_1600_SWA_subjects.txt
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)

if [ -e ${MRIS}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz ]; then

    python -u post_fmriprep_preproc_SRPBS_1600_SWA.py $sub
else
    echo ERROR: No func file found for subject $sub: ${MRIS}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz.
fi