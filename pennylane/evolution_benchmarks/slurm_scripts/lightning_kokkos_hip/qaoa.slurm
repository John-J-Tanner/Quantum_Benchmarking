#!/bin/bash --login
#SBATCH --account=pawsey0309-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --sockets-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --export=ALL

module load gcc
module load rocm/5.4.3
module load python/3.9.15

repeat=$1

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

SCRIPT_DIR=$(dirname "$SCRIPT_PATH" )

. $SCRIPT_DIR/../../../defaults.sh

VENV=gpu

# Check environment, activate Python venv.
. $SCRIPT_DIR/../setup.sh

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of threads, in this case will be 32
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
export SLURM_NTASKS=$SLURM_NTASKS
echo "SLURM_NTASKS $SLURM_NTASKS"
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo "SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"
export SLURM_GPUS=1
echo "SLURM_GPUS $SLURM_GPUS"


bash "$SCRIPT_DIR"/../../evolution_benchmarks.sh -l 2 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i $BENCHMARK_RUN_ID -o "$SCRIPT_DIR"/../../results -m qaoa_maxcut -a  qaoa_hypercube_maxcut_evolution -b lightning.kokkos -P python
