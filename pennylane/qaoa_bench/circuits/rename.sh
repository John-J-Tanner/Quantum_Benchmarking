for file in *.qasm; do
	mv $file lightning_kokkos_hip_${file}
done       
