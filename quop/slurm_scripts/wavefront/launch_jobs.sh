SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"
. $SCRIPT_DIR/../../defaults.sh

for repeat in $(seq 2 3); do
	#sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_qmoa_qft_lightning_kokkos.out  $SCRIPT_DIR/qmoa_qft.slurm $repeat $SCRIPT_DIR
	#sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_qmoa_hs_lightning_kokkos.out  $SCRIPT_DIR/qmoa_hs.slurm $repeat $SCRIPT_DIR
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_8_1_1_qaoa_wavefront.out  $SCRIPT_DIR/qaoa_1.slurm $repeat $SCRIPT_DIR
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_64_1_8_qaoa_wavefront.out  $SCRIPT_DIR/qaoa_8.slurm $repeat $SCRIPT_DIR
	#sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_grovers_lightning_kokkos.out $SCRIPT_DIR/grovers.slurm $repeat $SCRIPT_DIR
done

