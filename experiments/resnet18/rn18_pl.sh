#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --array=1-10%10
#SBATCH -p glinda

SAVEDIR=pl
RESNET="resnet18.py --optimizer powerlaw --total_epochs 40 --learn_rate 2e-3"

R1="--phi 0.1 --omega 1.0"
T1="--grad_glass --hess_comp abs"

NJOB=0
for SEED in {1..10}; do
    for REX in {1..1}; do
        for TEX in {1..1}; do
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

