#!/bin/bash

#SBATCH --job-name=KMMM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=25:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:8

module purge

singularity exec --nv \
	    --overlay /scratch/rw2867/envs/kmm.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate kmmm; torchrun --nproc_per_node=8 --master_port=1234\
                    /scratch/rw2867/projects/KMax-Mamba/train.py --data_root_path /scratch/rw2867/datasets/ --dist True --num_workers 8 --num_samples 1 --stage I --dataset_list PAOT_KMax TTS"