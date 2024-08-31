#!/bin/bash

#SBATCH --job-name=KMMM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=36GB
#SBATCH --gres=gpu:4

module purge

singularity exec --nv \
	    --overlay /scratch/rw2867/envs/kmm.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate kmmm; CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 --master_port=1234\
                    /scratch/rw2867/projects/KMax-Mamba/train.py --data_root_path /scratch/rw2867/datasets/ --dist True --dataset_list TTS --num_workers 4 --num_samples 2 --cache_dataset --cache_rate 0.1 --uniform_sample"