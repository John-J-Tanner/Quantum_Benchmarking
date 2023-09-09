#!/bin/bash --login
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

export SLURM_CPUS_PER_TASK=$1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores   
export OMP_PROC_BIND=close 

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

. "$SCRIPT_DIRECTORY"/../venvs/cpu/bin/activate


mkdir -p $SCRIPT_DIRECTORY/circuits
mkdir -p $SCRIPT_DIRECTORY/results

output_csv=${SCRIPT_DIRECTORY}/results/${SLURM_JOB_ID}_${SLURM_CPUS_PER_TASK}_lightning_kokkos_openmp.csv

echo "wall_time,expval,qubits,depth,repeats,backend,graph_sparsity,simulation_time" > $output_csv

seed=0
for qubit in ${qubits[@]}; do
	for depth in ${depths[@]}; do
		for sparse in ${sparsity[@]}; do
			output_qasm=${SCRIPT_DIRECTORY}/circuits/${SLURM_JOB_ID}_${qubit}_${depth}_$(echo $sparse | tr . _)_lightning_kokkos_openmp.qasm
			start_time=$(date +%s.%N)
			output=$(srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -m benchmark.qaoa_maxcut -q $qubit -d $depth -r 1 -g $sparse -b lightning.kokkos -c $output_qasm -s $seed)
			end_time=$(date +%s.%N)
			n_output=$(echo "$output" | grep -o "," | wc -l)
			if [ $n_output -eq 6 ]; then
				echo $(bc <<< "$end_time - $start_time"),$output >> $output_csv
			fi
			sleep 0.5
			seed=$(( $seed + 1 ))
		done
	done

done
