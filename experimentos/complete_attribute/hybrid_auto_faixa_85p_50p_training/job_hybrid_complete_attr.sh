#!/bin/bash
#SBATCH -p 1d
#SBATCH -t 1-00:00:00
#SBATCH --job-name=cpt_att_hybrid
#SBATCH -o hybrid_complete_attribute.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cristianoantonio.souza10@gmail.com
#SBATCH -n 1
#SBATCH --cpus-per-task 1
export PATH="/home/wzalewski/anaconda3/bin:$PATH"
source activate py27tensorflow
srun python main_hybrid_completeattribute.py
