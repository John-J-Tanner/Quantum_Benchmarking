SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"

OUTPUT_DIR="$SCRIPT_DIR"/../../slurm_output
mkdir -p "$OUTPUT_DIR"

for cpus_per_task in ${cpus[@]}; do
	for repeat in $(seq 1 3); do
		sbatch --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qmoa_qft_qiskit_aer.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_qft.slurm $repeat
		sbatch --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qmoa_hs_qiskit_aer.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_hs.slurm $repeat
		sbatch --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qaoa_qiskit_aer.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qaoa.slurm $repeat
		sbatch --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_grovers_qiskit_aer.out --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task $SCRIPT_DIR/grovers.slurm $repeat

	done
done

cpus_per_task=128
for repeat in $(seq 1 3); do
	sbatch --exclusive --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qmoa_qft_qiskit_aer.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_qft.slurm $repeat
	sbatch --exclusive --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qmoa_hs_qiskit_aer.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qmoa_hs.slurm $repeat
	sbatch --exclusive --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_qaoa_qiskit_aer.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/qaoa.slurm $repeat
	sbatch --exclusive --output=$OUTPUT_DIR/${repeat}_1_1_${cpus_per_task}_0_grovers_qiskit_aer.out --cpus-per-task=$cpus_per_task $SCRIPT_DIR/grovers.slurm $repeat

done

