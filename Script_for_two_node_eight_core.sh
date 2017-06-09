#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
module load Python/3.4.3-goolf-2015a

srun python twitter.py