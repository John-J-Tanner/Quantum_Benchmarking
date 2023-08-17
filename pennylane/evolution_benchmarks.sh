# Whitespace delimited list of Pennylane backends, e.g. "default.qubit lightning.qubit".
backends="$1"
# Directory in which to output benchmark results.
output_dir="$2"

# Time-limit (seconds) applied to each combination of state-evolution benchmark function + simulation backend.
benchmark_time_limit=3600
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
# Lower and upper bound for the number of simulated qubits.
qubits_min=4
qubits_max=30
# Number of sequential expectation value evaluations to carry out with randomly generated variational parameters.
# The total number of state propagations is depth*n_expval.
n_expval=100

echo Output directory: $output_dir
mkdir -p $output_dir

for backend in ${backends[@]}; do
	bench_start=$EPOCHSECONDS
	ansatz_index=0
	for ansatz in ${ansatze[@]}; do
		output_name=$output_dir/$(echo $backend | tr . _)_${ansatz}_$EPOCHSECONDS.csv
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
		#	func_time:		in-program time (seconds) taken call the state-evolution benchmark function.
		#	circuit_time:	in-program time (seconds) taken to carry out the state evolutions (excludes setup steps)
		#	wall_time:		program wall time
		#
		echo repeat,ansatz,backend,qubits,depth,n_expval,last_expval,func_time,circuit_time,wall_time >$output_name
		for qubits in $(seq $qubits_min $qubits_max); do
			for depth in ${depths[@]}; do
				for repeat in $(seq 1 $n_repeats); do
					echo Running repeat $repeat with $backend backend for $ansatz evolution with $qubits qubits at $depth depth...
					start=$EPOCHREALTIME
					output=$(python3 evolution_benchmark.py $backend ${ansatz_modules[$ansatz_index]} $ansatz $qubits $depth $n_expval ${ansatz_args[$ansatz_index]})
					end=$EPOCHREALTIME
					# If the output isn't a comma delimited list with three values, don't record the result and
					# move on to the next ansatz depth. This assumes that the simulation can't be performed with
					# the requested number of qubits.
					n_outvals=$(echo $output | grep -o ',' | wc -l)
					if [[ $n_outvals -eq 2 ]]; then
						echo $repeat,$ansatz,$backend,$qubits,$depth,$n_expval,$output,$(bc <<<"$end - $start") >>$output_name
						echo Done.
					else
						echo Skipping tests for $qubits qubits.
						break 2
					fi
					# If the benchmark time limit has been exceeded, move on to the next ansatz.
					if [[ $(($EPOCHSECONDS - $bench_start)) -gt $benchmark_time_limit ]]; then
						break 3
					fi
				done
			done
		done
		ansatz_index=$(($ansatz_index + 1))
	done
done
