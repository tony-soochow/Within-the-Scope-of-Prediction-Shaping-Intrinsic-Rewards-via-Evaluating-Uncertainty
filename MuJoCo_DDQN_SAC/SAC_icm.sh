#!/bin/bash
# Script to reproduce results
#for ((i=0;i<10;i+=1))

python -u run.py \
	--policy "Gaussian" \
        --filename "SAC-ICM_HalfCheetah_22.csv"   \
        --seed    "22"

python -u run.py \
	--policy "Gaussian" \
        --filename "SAC-ICM_HalfCheetah_23.csv"   \
        --seed    "23"

python -u run.py \
	--policy "Gaussian" \
        --filename "SAC-ICM_HalfCheetah_24.csv"   \
        --seed    "24"

python -u run.py \
	--policy "Gaussian" \
        --filename "SAC-ICM_HalfCheetah_25.csv"   \
        --seed    "25"

python -u run.py \
	--policy "Gaussian" \
        --filename "SAC-ICM_HalfCheetah_26.csv"   \
        --seed    "26"








