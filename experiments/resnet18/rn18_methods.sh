#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --array=1-100%20
#SBATCH -p glinda

SAVEDIR=methods
RESNET="resnet18.py --total_epochs 90 --reduce_epochs 80 --reduce_factor 0.1"

T1="--optimizer adam --learn_rate 1e-3"
T2="--optimizer sgdm --learn_rate 0.1"
T3="--optimizer adahessian --learn_rate 0.15"
T4="--optimizer alice --learn_rate 5e-3 --w1 0 --w2 0 --lr_min 0 --lr_max 1e-2 --grad_glass --hess_comp zero"
T5="--optimizer alice --learn_rate 5e-3 --w1 0 --w2 0 --lr_min 0 --lr_max 1e-2 --hess_comp abs"
T6="--optimizer alice --learn_rate 5e-3 --w1 0 --w2 0 --lr_min 0 --lr_max 1e-2 --grad_glass --hess_comp abs"
T7="--optimizer alice --learn_rate 5e-3 --w1 0 --w2 0 --lr_min 0 --lr_max 1e-2 --hess_comp rms"

NJOB=0
for SEED in {1..30}; do
    for EX in {1..7}; do
        ((NJOB++))
        VART=T$EX
        PREFIX=$SAVEDIR/T${EX}S${SEED}
        declare JOB$NJOB="python $RESNET ${!VART} --seed $SEED --log_prefix $PREFIX"
    done
done

. ../../venv_alice/bin/activate
VARJ=JOB${SLURM_ARRAY_TASK_ID}
srun ${!VARJ}

