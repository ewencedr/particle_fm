#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu 
#SBATCH --constraint=GPU
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --job-name fm_training
#SBATCH --output /beegfs/desy/user/birkjosc/epic-fm/logs/slurm_output/%x_%j.log      # terminal output
#SBATCH --mail-user joschka.birk@uni-hamburg.de
 
source ~/.bashrc
cd /home/birkjosc/repositories/EPiC-FM/

singularity exec --nv -B /home -B /beegfs /beegfs/desy/user/birkjosc/singularity_images/pytorch-image-v0.0.8.img \
    bash -c "source /opt/conda/bin/activate && python src/train.py experiment=jetclass_dev"
