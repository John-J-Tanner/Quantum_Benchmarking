#!/bin/bash --login
#SBATCH --account=pawsey0309-gpu
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --export=ALL

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:/software/projects/pawsey0309/edric_matwiejew/hipfftND/lib:/software/projects/pawsey0309/edric_matwiejew/software/hipTT/lib:/software/projects/pawsey0309/edric_matwiejew/opt/hipfort/lib:$LD_LIBRARY_PATH

repeat=$1
SCRIPT_DIR=$2

. $SCRIPT_DIR/../../defaults.sh
. $SCRIPT_DIR/../../../../env_setonix.sh

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of threads, in this case will be 32
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
export SLURM_NTASKS=64
echo "SLURM_NTASKS $SLURM_NTASKS"
export SLURM_CPUS_PER_TASK=1
echo "SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"
export SLURM_GPUS=8
echo "SLURM_GPUS $SLURM_GPUS"


bash "$BENCHMARK_ROOT"/evolution_benchmarks.sh -l 10 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i $BENCHMARK_RUN_ID -o "$BENCHMARK_ROOT"/results -m qaoa_maxcut -a  qaoa_hypercube_maxcut_evolution -b wavefront -P python3
