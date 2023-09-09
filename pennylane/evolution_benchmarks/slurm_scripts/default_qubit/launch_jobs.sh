SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"

OUTPUT_DIR="$SCRIPT_DIR"/../../slurm_output
mkdir -p "$OUTPUT_DIR"

for repeat in $(seq 1 1); do
	sbatch --output=$OUTPUT_DIR/${repeat}_1_1_1_0_qmoa_qft_default_qubit.out $SCRIPT_DIR/qmoa_qft.slurm $repeat 
	#sbatch --output=$OUTPUT_DIR/${repeat}_1_1_1_0_qmoa_hs_default_qubit.out $SCRIPT_DIR/qmoa_hs.slurm ${repeat} 
	#sbatch --output=$OUTPUT_DIR/${repeat}_1_1_1_0_qaoa_default_qubit.out $SCRIPT_DIR/qaoa.slurm ${repeat} 
	#sbatch --output=$OUTPUT_DIR/${repeat}_1_1_1_0_grovers_default_qubit.out $SCRIPT_DIR/grovers.slurm ${repeat} 
done
