SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"

cpus=(1 2 4 8 16 32 64)

for cpus_per_task in ${cpus[@]}; do
	sbatch --mem=$((2000 * $cpus_per_task)) --cpus-per-task=$cpus_per_task "$SCRIPT_DIR"/lightning_qubit.slurm $cpus_per_task
done

sbatch --exclusive --cpus-per-task=128 "$SCRIPT_DIR"/lightning_qubit.slurm 128

