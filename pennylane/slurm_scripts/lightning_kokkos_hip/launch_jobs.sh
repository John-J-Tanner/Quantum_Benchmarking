SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
. $SCRIPT_DIR/../../defaults.sh

for repeat in $(seq 1 3); do
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_qmoa_qft_lightning_kokkos.out  $SCRIPT_DIR/qmoa_qft.slurm $repeat
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_qmoa_hs_lightning_kokkos.out  $SCRIPT_DIR/qmoa_hs.slurm $repeat
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_qaoa_lightning_kokkos.out  $SCRIPT_DIR/qaoa.slurm $repeat
	sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_8_1_grovers_lightning_kokkos.out $SCRIPT_DIR/grovers.slurm $repeat
done

