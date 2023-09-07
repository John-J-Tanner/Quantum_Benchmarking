#!/bin/bash --login
#SBATCH --account=pawsey0309-gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --export=ALL


repeat=$1
SCRIPT_DIR=$2

. $SCRIPT_DIR/../../defaults.sh
. $SCRIPT_DIR/../../../../env_setonix.sh

module load rocm
module load craype-accel-amd-gfx90a

export MPICH_GPU_SUPPORT_ENABLED=1
export CXX=hipcc
export WAVEFRONT_BACKEND=ON
export HIPTT_PATH=$MYSOFTWARE/software/hipTT 
export HIPFORT_PATH=$MYSOFTWARE/opt/hipfort 
export HIPFFTND_PATH=$MYSOFTWARE/hipfftND 
export HIP_PLATFORM=amd
export MPI_GTL_LIB_DIR=${CRAY_MPICH_ROOTDIR}/gtl/lib
export HIPFORT_COMPILER=ftn
export OFFLOAD_ARCH=gfx90a

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:/software/projects/pawsey0309/edric_matwiejew/hipfftND/lib:/software/projects/pawsey0309/edric_matwiejew/software/hipTT/lib:/software/projects/pawsey0309/edric_matwiejew/opt/hipfort/lib:$LD_LIBRARY_PATH

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of threads, in this case will be 32
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
export SLURM_NTASKS=8
echo "SLURM_NTASKS $SLURM_NTASKS"
export SLURM_CPUS_PER_TASK=1
echo "SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"
export SLURM_GPUS=1
echo "SLURM_GPUS $SLURM_GPUS"


bash "$BENCHMARK_ROOT"/evolution_benchmarks.sh -l 10 -h 28 -n 100 -r $repeat -t $(( 12 * 60 * 60 )) -i $BENCHMARK_RUN_ID -o "$BENCHMARK_ROOT"/results -m qaoa_maxcut -a  qaoa_hypercube_maxcut_evolution -b wavefront -P python3
