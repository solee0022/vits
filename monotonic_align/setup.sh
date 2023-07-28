#!/bin/bash

# sbatch
#SBATCH -J setup # job name
#SBATCH -o %x_%j.out
#SBATCH -p 3090 #queue name or patition name
#SBATCH -t 72:00:00 # Run time

# gpu 설정
## gpu 개수
#SBATCH   --gres=gpu:1
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=1

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

echo "CUDA_HOME=$CUDA_HOME"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname

srun -l /bin/pwd

srun -l /bin/date

echo "START"

python /home/solee0022/tts-asr/XPhoneBERT/VITS_with_XPhoneBERT/monotonic_align/setup.py build_ext --inplace

data
echo "END"