#!/bin/bash --login
### INPUT ARGUMENTS
echo "Parsed input arguments:"
while getopts 'l:h:n:r:t:i:o:m:a:p:b:P:' opt; do
	case $opt in
	l)
		qubits_min=$OPTARG
		printf "\tMin qubits: $qubits_min\n"
		;;
	h)
		qubits_max=$OPTARG
		printf "\tMax qubits: $qubits_max\n"
		;;
	n)
		n_expval=$OPTARG
		printf "\tCircuit evaluations: $n_expval\n"
		;;
	r)
		repeat_num=$OPTARG
		printf "\tRepeat number: $repeat_num\n"
		;;
	t)
		benchmark_time_limit=$OPTARG
		printf "\tTime limit (s): $benchmark_time_limit\n"
		;;
	i)
		benchmark_set_ID=$OPTARG
		printf "\tBenchmark set ID: $benchmark_set_ID\n"
		;;
	o)
		results_rootdir=$OPTARG
		printf "\tResults root directory: $results_rootdir\n"
		;;
	m)
		ansatz_module=$OPTARG
		printf "\tAnsatz module: $ansatz_module\n"
		;;
	a)
		ansatz=$OPTARG
		printf "\tAnsatz function: $ansatz\n"
		;;
	p)
		ansatz_args=$OPTARG
		printf "\tAdditional ansatz function arguments: $ansatz_args\n"
		;;
	b)
		backend="$OPTARG"
		printf "\tSimulation backend: $backend\n"
		;;
	P)
		python_executable=$OPTARG
		;;
	esac
done

python_executable=${python_executable=python}
printf "\tPython executable: $python_executable\n"

printf "\nSystem configuration:\n"
# Number of GPUS, assumed to be zero if $SLURM_GPUS is not defined.
NGPUS=${SLURM_GPUS=0}
printf "\tNGPUS: $SLURM_GPUS\n"
# Number of nodes, if $SLURM_NNODES is undefined, do not launch with Slurm.
NNODES=${SLURM_NNODES=0}
printf "\tNNODES: $NNODES\n"

case $NNODES in
0)
	export OMP_NUM_THREADS=$(( $(grep -m 1 'cpu cores' '/proc/cpuinfo' | cut -d : -f 2) ))
	CPUS_PER_TASK=$OMP_NUM_THREADS
	NTASKS=0
	;;
*)
	export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
	CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
	NTASKS=$SLURM_NTASKS
	;;
esac
printf "\tNTASKS: $NTASKS\n"
printf "\tCPUS_PER_TASK: $CPUS_PER_TASK\n"

output_dir="$results_rootdir"/"$benchmark_set_ID"
output_name=$output_dir/${repeat_num}_${NNODES}_${NTASKS}_${CPUS_PER_TASK}_${NGPUS}_$(echo $backend | tr . _)_${ansatz}.csv
output_suspend_name=$output_dir/${repeat_num}_${NNODES}_${NTASKS}_${CPUS_PER_TASK}_${NGPUS}_$(echo $backend | tr . _)_${ansatz}_suspend.txt
mkdir -p "$output_dir"
printf "\nOutput path: $output_name\n"

# A different output file is used for each combination of backend and algorithm ansatz.
#
# Header for results file:
#
# 	repeats:	 		repeat number
# 	ansatz: 			name of the state-evolution benchmark function
# 	backend:	 		Pennylane simulation backend used
# 	qubits: 			number of simulated qubits
#	depth: 				simulated circuit depth
#	n_expval:  			number of expectation value evaluations
#	last_expval:		last expectation value, should be the same for all simulation backends
#	func_time:			in-program time (seconds) taken call the state-evolution benchmark function
# 	gate_depth:			circuit depth in terms of gates 
#	one_qubit_gates:	number of one-qubit gates (e.g. Pauli gates)
#	two_qubit_gates:	number of two-qubit gates (e.g. CNOT)
#	circuit_time:		in-program time (seconds) taken to carry out the state evolutions (excludes setup steps)
#	wall_time:			program wall time
#	n_nodes:			number of nodes (SLURM_NNODES)
#	n_cpus:				number of CPUs per task (SLURM_CPUS_PER_TASK)
#	n_gpus:				number of GPUS (SLURM_GPUS)
#

echo repeat,ansatz,backend,qubits,depth,n_expval,last_expval,func_time,circuit_time,wall_time,n_nodes,n_cpus,n_gpus,one_qubit_gates,two_qubit_gates >$output_name

bench_start=$(date +%s)
time_remaining=$benchmark_time_limit

depths=(1 2 4 8 16 32)

for qubits in $(seq $qubits_min $qubits_max); do
	for depth in ${depths[@]}; do
		launch_command="$python_executable evolution_benchmark.py $backend $ansatz_module $ansatz $qubits $depth $repeat_num $n_expval $ansatz_args"

		time_percent_remaining=$(bc <<<"scale=1;100 - 100*($(date +%s.%N)- $bench_start)/$benchmark_time_limit")
		echo "(Benchmark time remaining: $time_percent_remaining%)": Running repeat $repeat_num with $backend backend for $ansatz evolution with $qubits qubits at depth $depth.

		start_time=$(date +%s.%N)
		case $NNODES in
		0)
			output=$($launch_command)
			;;
		*)
			if [ $NGPUS -gt 0 ]; then
				output=$(srun -N $NNODES -n $NTASKS -c $CPUS_PER_TASK --gpus=$NGPUS $launch_command)
			else
				output=$(srun -N $NNODES -n $NTASKS -c $CPUS_PER_TASK $launch_command)
			fi
			;;
		esac
		end_time=$(date +%s.%N)

		status=$(echo "$output" | cut -d, -f 1)
		results=$(echo "$output" | cut -d, -f 2-)

		case $status in
		success)
			echo $repeat_num,$ansatz,$backend,$qubits,$depth,$n_expval,$results,$(bc <<<"$end_time - $start_time"),$NNODES,$CPUS_PER_TASK,$NGPUS >>$output_name
			;;
		pass)
			echo "Skipping tests for $qubits qubits."
			break 1
			;;
		*)
			echo "Test at $depth depth with $qubits qubits failed, concluding tests for $ansatz with $backend backend."
			break 2
			;;
		esac

		time_remaining=$(bc <<<"$time_remaining - ($end_time - $start_time)")
		time_test=$(bc <<<"($time_remaining - ($end_time - $start_time))/1")

		if [ $time_test -le 0 ]; then
			echo "Not enough time for next test, concluding tests for $ansatz with $backend backend."
			printf "Suspended benchmark at:\n" >$output_suspend_name
			printf "qubits:$qubits\ndepth:$depth\n" >>$output_suspend_name
			break 2
		fi
	done

done
