#!/bin/bash
#SBATCH --nodelist=chacha
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

apptainer exec --nv \
    --bind $PHOTOCAST_DATA_DIR/images:$PWD/data/images:ro \
    --bind $PHOTOCAST_OUTDIR/runs:$PWD/results \
    $HOME/.apptainer/jit.sif \
    $1