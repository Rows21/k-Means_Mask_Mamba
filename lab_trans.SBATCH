#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu
#SBATCH --job-name=KMMM

module purge

singularity exec --nv \
	    --overlay /scratch/rw2867/envs/kmm.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate kmmm; python /scratch/rw2867/projects/KMax-Mamba/label_transfer.py"