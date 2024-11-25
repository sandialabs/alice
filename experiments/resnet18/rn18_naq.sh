#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --array=1-80%20
#SBATCH -p glinda

SAVEDIR=naq
RESNET="resnet18.py --optimizer alice --learn_rate 5e-3 --total_epochs 60 --limit_method fixed --lr_min 0 --lr_max 5e-3"

R1="--phi 1.0 --omega 1.0"
R2="--phi 0.1 --omega 1.0"

T1="--grad_glass --hess_comp zero"
T2="--hess_comp abs"
T3="--grad_glass --hess_comp abs"
T4="--hess_comp rms"

NJOB=0
for SEED in {1..10}; do
    for REX in {1..2}; do
        for TEX in {1..4}; do
            ((NJOB++))
            VARR=R$REX
            VART=T$TEX
            PREFIX=$SAVEDIR/R${REX}T${TEX}S${SEED}
            declare JOB$NJOB="python $RESNET ${!VARR} ${!VART} --seed $SEED --log_prefix $PREFIX"
        done
    done
done

. ../../venv_alice/bin/activate
VARJ=JOB${SLURM_ARRAY_TASK_ID}
srun ${!VARJ}

