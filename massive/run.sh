#!/bin/bash
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:1
#SBATCH --mail-user=utkarsh.upadhyay@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new/hi_hf
#SBATCH --time=72:00:00

module load u18/ffmpeg/5.0.1
eval "$(conda shell.bash hook)"
conda activate IS

mkdir -p logs/new/en_orig
python train.py --lang en --accent orig --test_accent tamil_female --cpu_cores 8 > logs/new/en_orig/tamil_female 2>&1
python train.py --lang en --accent orig --test_accent bengali_female --cpu_cores 8 --mode test > logs/new/en_orig/bengali_female 2>&1
python train.py --lang en --accent orig --test_accent malayalam_male --cpu_cores 8 --mode test > logs/new/en_orig/malayalam_male 2>&1
python train.py --lang en --accent orig --test_accent manipuri_female --cpu_cores 8 --mode test > logs/new/en_orig/manipuri_female 2>&1
python train.py --lang en --accent orig --test_accent assamese_female --cpu_cores 8 --mode test > logs/new/en_orig/assamese_female 2>&1
python train.py --lang en --accent orig --test_accent gujarati_male --cpu_cores 8 --mode test > logs/new/en_orig/gujarati_male 2>&1
python train.py --lang en --accent orig --test_accent telugu_male --cpu_cores 8 --mode test > logs/new/en_orig/telugu_male 2>&1
python train.py --lang en --accent orig --test_accent kannada_male --cpu_cores 8 --mode test > logs/new/en_orig/kannada_male 2>&1
python train.py --lang en --accent orig --test_accent hindi_female --cpu_cores 8 --mode test > logs/new/en_orig/hindi_female 2>&1
python train.py --lang en --accent orig --test_accent rajasthani_female --cpu_cores 8 --mode test > logs/new/en_orig/rajasthani_female 2>&1
python train.py --lang en --accent orig --test_accent kannada_female --cpu_cores 8 --mode test > logs/new/en_orig/kannada_female 2>&1
python train.py --lang en --accent orig --test_accent bengali_male --cpu_cores 8 --mode test > logs/new/en_orig/bengali_male 2>&1
python train.py --lang en --accent orig --test_accent tamil_male --cpu_cores 8 --mode test > logs/new/en_orig/tamil_male 2>&1
python train.py --lang en --accent orig --test_accent gujarati_female --cpu_cores 8 --mode test > logs/new/en_orig/gujarati_female 2>&1
python train.py --lang en --accent orig --test_accent assamese_male --cpu_cores 8 --mode test > logs/new/en_orig/assamese_male 2>&1
python train.py --lang en --accent orig --test_accent orig --cpu_cores 8 --mode test > logs/new/en_orig/orig 2>&1


# mkdir -p logs/new/hi_orig
# python train.py --lang hi --accent orig --test_accent bengali_female --cpu_cores 8 > logs/new/hi_orig/bengali_female 2>&1
# python train.py --lang hi --accent orig --test_accent bengali_male --cpu_cores 8 --mode test > logs/new/hi_orig/bengali_male 2>&1
# python train.py --lang hi --accent orig --test_accent gujarati_female --cpu_cores 8 --mode test > logs/new/hi_orig/gujarati_female 2>&1
# python train.py --lang hi --accent orig --test_accent gujarati_male --cpu_cores 8 --mode test > logs/new/hi_orig/gujarati_male 2>&1
# python train.py --lang hi --accent orig --test_accent hindi_female --cpu_cores 8 --mode test > logs/new/hi_orig/hindi_female 2>&1
# python train.py --lang hi --accent orig --test_accent hindi_male --cpu_cores 8 --mode test > logs/new/hi_orig/hindi_male 2>&1
# python train.py --lang hi --accent orig --test_accent kannada_female --cpu_cores 8 --mode test > logs/new/hi_orig/kannada_female 2>&1
# python train.py --lang hi --accent orig --test_accent kannada_male --cpu_cores 8 --mode test > logs/new/hi_orig/kannada_male 2>&1
# python train.py --lang hi --accent orig --test_accent malayalam_female --cpu_cores 8 --mode test > logs/new/hi_orig/malayalam_female 2>&1
# python train.py --lang hi --accent orig --test_accent malayalam_male --cpu_cores 8 --mode test > logs/new/hi_orig/malayalam_male 2>&1
# python train.py --lang hi --accent orig --test_accent manipuri_female --cpu_cores 8 --mode test > logs/new/hi_orig/manipuri_female 2>&1
# python train.py --lang hi --accent orig --test_accent rajasthani_female --cpu_cores 8 --mode test > logs/new/hi_orig/rajasthani_female 2>&1
# python train.py --lang hi --accent orig --test_accent rajasthani_male --cpu_cores 8 --mode test > logs/new/hi_orig/rajasthani_male 2>&1
# python train.py --lang hi --accent orig --test_accent tamil_female --cpu_cores 8 --mode test > logs/new/hi_orig/tamil_female 2>&1
# python train.py --lang hi --accent orig --test_accent tamil_male --cpu_cores 8 --mode test > logs/new/hi_orig/tamil_male 2>&1
# python train.py --lang hi --accent orig --test_accent telugu_female --cpu_cores 8 --mode test > logs/new/hi_orig/telugu_female 2>&1
# python train.py --lang hi --accent orig --test_accent telugu_male --cpu_cores 8 --mode test > logs/new/hi_orig/telugu_male 2>&1
# python train.py --lang hi --accent orig --test_accent orig --cpu_cores 8 --mode test > logs/new/hi_orig/orig 2>&1

