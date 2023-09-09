#!/bin/bash --login
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

module load gcc
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

VENV=cpu

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

bash "$SCRIPT_DIR"/../../evolution_benchmarks.sh -l 10 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i $BENCHMARK_RUN_ID -o "$SCRIPT_DIR"/../../results -m qmoa_styblinski_tang -a qmoa_complete_ST_evolution_QFT -b qiskit.aer -P python -p 4
