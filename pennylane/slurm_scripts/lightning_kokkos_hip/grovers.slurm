#!/bin/bash --login
#SBATCH --account=pawsey0309-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --sockets-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

module load gcc
module load rocm/5.4.3
module load python/3.9.15

repeat=$1
SCRIPT_DIR=$2

. $SCRIPT_DIR/../../defaults.sh

VENV=$GPU_VENV

# Check environment, activate Python venv.
. $SCRIPT_DIR/../setup.sh

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores   
export OMP_PROC_BIND=close 
 
# Exporting environment variables required in evolution_benchmarks.sh.
export SLURM_NTASKS=$SLURM_NTASKS
echo "SLURM_NTASKS $SLURM_NTASKS"
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo "SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"
export SLURM_GPUS=1
echo "SLURM_GPUS $SLURM_GPUS"

bash evolution_benchmarks.sh -l 3 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i $BENCHMARK_RUN_ID -o "$BENCHMARK_ROOT"/results -m grovers_search -a grovers_search_decomposed -b lightning.kokkos -P python
