SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
. $SCRIPT_DIR/../../defaults.sh

repeats=(1 2 3)

for repeat in ${repeats[@]}; do
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_1_0_qmoa_qft_default_qubit.out $SCRIPT_DIR/qmoa_qft.slurm ${repeat}
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_1_0_qmoa_hs_default_qubit.out $SCRIPT_DIR/qmoa_hs.slurm ${repeat}
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_1_0_qaoa_default_qubit.out $SCRIPT_DIR/qaoa.slurm ${repeat}
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_1_0_grovers_default_qubit.out $SCRIPT_DIR/grovers.slurm ${repeat}
done
