### INPUT ARGUMENTS 

# Whitespace delimited list of Pennylane backends, e.g. "default.qubit lightning.qubit".
backends="$1"
# Directory in which to output benchmark results.
results_rootdir="$2"
# Subfolder to seperate benchmark results (dates, different configs and so on).
benchmark_set_ID="$3"
# Time-limit (seconds) applied to each combination of state-evolution benchmark function + simulation backend.
benchmark_time_limit=$4
# Lower and upper bound for the number of simulated qubits.
qubits_min=$5
qubits_max=$6

####################

# Number of repeats.
n_repeats=3
# Modules containing state evolution benchmark functions.
ansatz_modules=(qmoa_styblinski_tang qmoa_styblinski_tang qaoa_maxcut qaoa_maxcut)
# State-evolution benchmark functions.
ansatze=(qmoa_complete_ST_evolution_QFT qmoa_complete_ST_evolution_HS qaoa_complete_maxcut_evolution qaoa_hypercube_maxcut_evolution)
# Additional arguments for each state-evolution benchmark function.
ansatz_args=(4 4 "" "")
# Simulated ansatz iterations.
depths=(1 2 4 8 16 32 64)
# Number of sequential expectation value evaluations to carry out with randomly generated variational parameters.
# The total number of state propagations is depth*n_expval.
n_expval=100

output_dir="$results_rootdir"/$benchmark_set_ID
mkdir -p "$output_dir"
echo Output directory: "$output_dir"

# Number of GPUS, assumed to be zero if $SLURM_GPUS is not defined.
N_GPUS=${SLURM_GPUS+0}
echo N_GPUS: $N_GPUS

for backend in ${backends[@]}; do
	ansatz_index=0
	for ansatz in ${ansatze[@]}; do
		# End the benchmark for backend if $(date +%s) - $bench_start > $benchmark_time_limit.
		bench_start=$(date +%s)

		output_name=$output_dir/$(echo $backend | tr . _)_${ansatz}.csv
		# A different output file is used for each combination of backend and algorithm ansatz.
		#
		# Header for results file:
		#
		# 	repeats:	 	repeat number
		# 	ansatz: 		name of the state-evolution benchmark function
		# 	backend:	 	Pennylane simulation backend used
		# 	qubits: 		number of simulated qubits
		#	depth: 			simulated circuit depth
		#	n_expval:  		number of expectation value evaluations
		#	last_expval:	last expectation value, should be the same for all simulation backends
		#	func_time:		in-program time (seconds) taken call the state-evolution benchmark function
		#	circuit_time:	in-program time (seconds) taken to carry out the state evolutions (excludes setup steps)
		#	wall_time:		program wall time
		#	n_nodes:		number of nodes (SLURM_NNODES)
		#	n_cpus:			number of CPUs per task (SLURM_CPUS_PER_TASK)
		#	n_gpus:			number of GPUS (SLURM_GPUS)
		#
		echo repeat,ansatz,backend,qubits,depth,n_expval,last_expval,func_time,circuit_time,wall_time,n_nodes,n_cpus,n_gpus >$output_name
		for qubits in $(seq $qubits_min $qubits_max); do
			for depth in ${depths[@]}; do
				for repeat in $(seq 1 $n_repeats); do

					time_remaining=$(bc <<< "scale=1;100 - 100*($(date +%s.%N)- $bench_start)/$benchmark_time_limit")
					echo "(Benchmark time remaining: $time_remaining%)": Running repeat $repeat with $backend backend for $ansatz evolution with $qubits qubits at depth $depth.

					start=$(date +%s.%N)
					if [ $N_GPUS -gt 0 ]; then
						output=$(srun -N $SLURM_NNODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --gpus=$SLURM_GPUS python evolution_benchmark.py $backend ${ansatz_modules[$ansatz_index]} $ansatz $qubits $depth $n_expval $repeat ${ansatz_args[$ansatz_index]}) 
					else
						output=$(srun -N $SLURM_NNODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK python evolution_benchmark.py $backend ${ansatz_modules[$ansatz_index]} $ansatz $qubits $depth $n_expval $repeat ${ansatz_args[$ansatz_index]})
					fi
					end=$(date +%s.%N)

					status=$(echo "$output" | cut -d, -f 1)	
					results=$(echo "$output" | cut -d, -f 2- )	

					case $status in
						success)
							echo $repeat,$ansatz,$backend,$qubits,$depth,$n_expval,$results,$(bc <<<"$end - $start"),$SLURM_NNODES,$SLURM_CPUS_PER_TASK,$N_GPUS >>$output_name
							;;
						pass)
							echo "Skipping tests for $qubits qubits."
							break 2
							;;
						*)
						 	echo "Test at $depth depth with $qubits qubits failed, concluding tests for $ansatz with $backend backend."
							break 3
							;;
					esac

					# If the benchmark time limit has been exceeded, move on to the next ansatz.
					if [[ $(( $(date +%s) - $bench_start)) -gt $benchmark_time_limit ]]; then
					 	echo "Benchmark time limit exceeded, moving to next ansatz."
						break 3
					fi

				done
			done
		done
		ansatz_index=$(($ansatz_index + 1))
	done
done
