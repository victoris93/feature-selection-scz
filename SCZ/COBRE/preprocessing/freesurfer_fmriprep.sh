#!/bin/bash

############################################################
############# BEGIN SLURM CONFIGURATION ######################

#SBATCH --job-name=fmriprep
#SBATCH -o ./logs/fmriprep-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array 2-74:1

############################################################
############## BEGIN SCRIPT TO RUN #########################

export FS_LICENSE=$HOME/freesurfer_license.txt
export SIF=/gpfs3/well/margulies/projects/fmriprep
SUBJECT_LIST=./COBRE_subjects.txt

# Define variables to point to the directories you want to work in
# This is not strictly necessary, but may be helpful
# bids and derivatives directories are siblings in this scheme unde a directory called MRI
export MRIS=/gpfs3/well/margulies/projects/data/COBRE

SUBJECT_LIST=./COBRE_subjects.txt
echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo the job id is $SLURM_ARRAY_JOB_ID

sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
# Just to be safe, create the derivatives subdirectory if it does not exist.
if [ ! -d ${MRIS}/freesurfer ]; then 
    mkdir ${MRIS}/derivatives
fi

# Just to be safe, create the work directory if it does not exist. Can't be in the BIDS input folder
if [ ! -d ${SIF}/fmriprep_work ]; then 
mkdir ${SIF}/fmriprep_work
fi


echo Starting fmriprep, subject $sub
singularity exec -B $MRIS -B $SIF $SIF/containers_bids-fmriprep--20.2.1.sif /usr/local/miniconda/bin/fmriprep \
 $MRIS $MRIS/derivatives participant \
 --participant_label $sub \
 --skip-bids-validation \
 --md-only-boilerplate \
 --fs-license-file $FS_LICENSE \
 --fs-no-reconall \
 --stop-on-first-crash \
 --output-spaces MNI152NLin2009cAsym:res-2 \
 -w ${SIF}/fmriprep_work
