echo "file,one_qubit_gates,two_qubit_gates" > gate_counts.csv
for file in *.qasm;
do
	one_qubit=$(awk '/^h|^u[1-4]|^rz/ {count++} END {print count}'  $file)
	two_qubit=$(awk '/^cx/ {count++} END {print count}' $file)
	echo $file,$one_qubit,$two_qubit >> gate_counts.csv
done

