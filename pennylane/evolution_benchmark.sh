backends="$1"
output_dir="$2"

benchmark_time_limit=3600
n_repeats=1
ansatze=(qmoa_complete_ST_evolution_HS qaoa_complete_maxcut_evolution qaoa_hypercube_maxcut_evolution )
ansatz_args=(4 "" "")
ansatz_modules=(qmoa_styblinski_tang qaoa_maxcut qaoa_maxcut)
depths=(1 2 4 8 16 32 64)
qubits_min=4
qubits_max=30

n_expval=100

echo Output directory: $output_dir
mkdir -p $output_dir

for backend in ${backends[@]}
do
	bench_start=$EPOCHSECONDS
	ansatz_index=0
	for ansatz in ${ansatze[@]}
	do
		output_name=$output_dir/$(echo $backend | tr . _)_${ansatz}_$EPOCHSECONDS.csv
		echo repeat,ansatz,backend,qubits,depth,n_expval,last_expval,func_time,circuit_time,wall_time > $output_name
		for qubits in $(seq $qubits_min $qubits_max) 
		do
			for depth in ${depths[@]}
			do
				for repeat in $(seq 1 $n_repeats)
				do
					echo Running repeat $repeat with $backend backend for $ansatz evolution with $qubits qubits at $depth depth...
					start=$EPOCHREALTIME
					output=$(python3 evolution_benchmark.py $backend ${ansatz_modules[$ansatz_index]} $ansatz $qubits $depth $n_expval ${ansatz_args[$ansatz_index]})
					end=$EPOCHREALTIME
					n_outvals=$(echo $output | grep -o ',' | wc -l)
					if [[ $n_outvals -eq 2 ]]
					then
						echo $repeat,$ansatz,$backend,$qubits,$depth,$n_expval,$output,$(bc <<< "$end - $start") >> $output_name
						echo Done.
					else
					 	echo Skipping tests for $qubits qubits.
						break 2
					fi
					if [[ $(( $EPOCHSECONDS - $bench_start )) -gt $benchmark_time_limit ]]
					then
						break 3
					fi
				done
			done
		done
		ansatz_index=$(( $ansatz_index + 1 ))	
	done
done
