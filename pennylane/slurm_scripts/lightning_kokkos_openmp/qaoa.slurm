#!/bin/bash --login
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

module load gcc
module load python/3.9.15

# Check environment, activate Python venv.
. setup.sh $1 $CPU_VENV

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores   
export OMP_PROC_BIND=close 
 
# Exporting environment variables required in evolution_benchmarks.sh.
export SLURM_NTASKS=$SLURM_NTASKS
echo "SLURM_NTASKS $SLURM_NTASKS"
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo "SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"

bash "$BENCHMARK_ROOT"/evolution_benchmarks.sh -l 2 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i 31-08-2023 -o "$BENCHMARK_ROOT"/results -m qaoa_maxcut -a  qaoa_hypercube_maxcut_evolution -b lightning.kokkos -P python
