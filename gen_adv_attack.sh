#!/bin/bash
#SBATCH --job-name=harmbench
#SBATCH --output=harmbench.out
#SBATCH --error=harmbench.err
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
source /home/hmanikan/anaconda3/bin/activate
cd /home/hmanikan/harmbench-dev
python -u vae_gumbel.py --train
# ssh -L 8085:localhost:8085 hmanikan@locus.cs.cmu.edu