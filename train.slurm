#!/bin/bash
#SBATCH --job-name=loraTest
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --gres=gpu:8                # number of gpus
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=ENT212162
#SBATCH --partition=gp4d

# init environment 
export PYTHONUSERBASE=$CONDA_PREFIX
export HF_HOME=/work/twsuzrf718/hf_home

# setup distrubuted environment
echo "NODELIST="$SLURM_JOB_NODELIST
export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
export OMP_NUM_THREADS=1

# generate configs for huggingface accelerator
export NNODES=$SLURM_NNODES
export TOTAL_GPUS=$((NNODES*8))
srun --jobid $SLURM_JOBID bash -c './generate_config.sh $SLURM_JOBID $SLURM_NODEID $MASTER_ADDR $MASTER_PORT $NNODES $TOTAL_GPUS'

# submit slurm job
export LAUNCHER="accelerate launch"

export CMD="peft_bit_clm_accelerate_ds_zero3_offload.py"

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --config_file config.$SLURM_JOBID.$SLURM_NODEID.yaml $CMD'
echo "Finish $(date)"
