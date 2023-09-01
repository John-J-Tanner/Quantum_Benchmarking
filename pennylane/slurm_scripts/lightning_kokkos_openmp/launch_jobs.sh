SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"
. $SCRIPT_DIR/../../defaults.sh

cpus=(1 16 32 64)

for cpus_per_task in ${cpus[@]}; do
	for repeat in $(seq 1 3); do
		sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qmoa_qft_lightning_kokkos.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_qft.slurm $repeat "$SCRIPT_DIR"
		sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qmoa_hs_lightning_kokkos.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_hs.slurm $repeat "$SCRIPT_DIR" 
		sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qaoa_lightning_kokkos.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qaoa.slurm $repeat "$SCRIPT_DIR"
		sbatch --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_grovers_lightning_kokkos.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/grovers.slurm $repeat "$SCRIPT_DIR"
	done
done

cpus_per_task=128
for repeat in $(seq 1 3); do
	sbatch --exclusive --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qmoa_qft_lightning_kokkos.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_qft.slurm $repeat "$SCRIPT_DIR"
	sbatch --exclusive --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qmoa_hs_lightning_kokkos.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_hs.slurm $repeat "$SCRIPT_DIR"
	sbatch --exclusive --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_qaoa_lightning_kokkos.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qaoa.slurm $repeat "$SCRIPT_DIR"
	sbatch --exclusive --output=$BENCHMARK_OUTPUT/${repeat}_1_1_${cpus_per_task}_0_grovers_lightning_kokkos.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/grovers.slurm $repeat "$SCRIPT_DIR"
done

