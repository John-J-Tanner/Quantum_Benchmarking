#!/bin/bash --login

repeat=$1

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
. $SCRIPT_DIR/../../defaults.sh 

#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=$((8000 * 1))M
#SBATCH --time=00:02:00
#SBATCH --output=$BENCHMARK_OUTPUT/default_qubit_${repeat}.out
#SBATCH --qos=high

module load gcc
module load python/3.9.15

. venv/cpu/bin/activate
 
# ---
# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of threads, in this case will be 32
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# ---
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

bash evolution_benchmarks.sh -q 2 -h 28 -n 100 -r $repeat -t $(( 2 * 60 )) -i 31-08-2023 -o $BENCHMARK_ROOT/results -m grovers_search -a grovers_search -b default.qubit -P python &