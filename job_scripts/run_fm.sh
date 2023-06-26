#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu 
#SBATCH --constraint=GPU
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --job-name fm_training
#SBATCH --output /beegfs/desy/user/birkjosc/epic-cnf/logs/%x_%j.log      # terminal output
#SBATCH --mail-user joschka.birk@uni-hamburg.de
 
source ~/.bashrc
 
# activate your conda environment the job should use
# go to your folder with your python scripts
cd /home/birkjosc/repositories/EPiC-FM/

singularity exec --nv -B /home -B /beegfs /beegfs/desy/user/birkjosc/singularity_images/pytorch-image-v0.0.8.img \
    bash -c "source /opt/conda/bin/activate && python src/train.py experiment=fm_tops_slurm"
