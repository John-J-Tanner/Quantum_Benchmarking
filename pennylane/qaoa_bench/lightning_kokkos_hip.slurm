#!/bin/bash --login
#SBATCH --account=pawsey0309-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --sockets-per-node=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

SCRIPT_DIRECTORY=$(dirname "$SCRIPT_PATH" )

qubits=(10 16 20 24)
depths=(1 4 16 64)
sparsity=(0.125 0.25 0.375 0.5)

. "$SCRIPT_DIRECTORY"/../venvs/gpu/bin/activate


mkdir -p $SCRIPT_DIRECTORY/circuits
mkdir -p $SCRIPT_DIRECTORY/results

output_csv=${SCRIPT_DIRECTORY}/results/${SLURM_JOB_ID}_lightning_kokkos_hip.csv

echo "expval,qubits,depth,repeats,backend,graph_sparsity,simulation_time,circuit_depth,one_qubit_gates,two_qubit_gates" > $output_csv

for qubit in ${qubits[@]}; do
	for depth in ${depths[@]}; do
		for sparse in ${sparsity[@]}; do
			output_qasm=${SCRIPT_DIRECTORY}/circuits/${SLURM_JOB_ID}_${qubits}_${depth}_$(echo $sprase | tr . _)_lightning_kokkos_hip.qasm
			output=$(srun -N 1 -n 1 -c 8 --gpus=1 --gpu-bind=closest python -m benchmark.qaoa_maxcut -q $qubit -d $depth -r 1 -G $sparse -b default.qubit -c $output_qasm)
			echo $output >> $output_csv
			sleep 0.5
		done
	done

done
