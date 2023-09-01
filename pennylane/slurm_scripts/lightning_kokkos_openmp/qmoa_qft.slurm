#!/bin/bash --login
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

module load gcc
module load python/3.9.15

repeat=$1
SCRIPT_DIR=$2

. $SCRIPT_DIR/../../defaults.sh

VENV=$CPU_VENV

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

bash "$BENCHMARK_ROOT"/evolution_benchmarks.sh -l 2 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i 31-08-2023 -o "$BENCHMARK_ROOT"/results -m qmoa_styblinski_tang -a qmoa_complete_ST_evolution_QFT -b lightning.kokkos -P python -p 4
