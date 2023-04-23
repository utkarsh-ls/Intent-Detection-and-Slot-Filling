#!/bin/bash
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:1
#SBATCH --mail-user=utkarsh.upadhyay@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=h_m_val.out
#SBATCH --time=72:00:00

module load u18/ffmpeg/5.0.1
eval "$(conda shell.bash hook)"
conda activate IS
python tts_stt.py
