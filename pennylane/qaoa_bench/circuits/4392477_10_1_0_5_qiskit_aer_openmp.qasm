OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
cx q[0],q[1];
u1(0.08564916714362436) q[1];
cx q[0],q[1];
cx q[0],q[3];
u1(0.08564916714362436) q[3];
cx q[0],q[3];
cx q[1],q[2];
u1(0.08564916714362436) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.08564916714362436) q[4];
cx q[1],q[4];
cx q[1],q[7];
u1(0.08564916714362436) q[7];
cx q[1],q[7];
cx q[1],q[8];
u1(0.08564916714362436) q[8];
cx q[1],q[8];
cx q[1],q[9];
u1(0.08564916714362436) q[9];
cx q[1],q[9];
cx q[2],q[3];
u1(0.08564916714362436) q[3];
cx q[2],q[3];
cx q[2],q[4];
u1(0.08564916714362436) q[4];
cx q[2],q[4];
cx q[2],q[5];
u1(0.08564916714362436) q[5];
cx q[2],q[5];
cx q[2],q[7];
u1(0.08564916714362436) q[7];
cx q[2],q[7];
cx q[2],q[8];
u1(0.08564916714362436) q[8];
cx q[2],q[8];
cx q[3],q[4];
u1(0.08564916714362436) q[4];
cx q[3],q[4];
cx q[3],q[5];
u1(0.08564916714362436) q[5];
cx q[3],q[5];
cx q[3],q[9];
u1(0.08564916714362436) q[9];
cx q[3],q[9];
cx q[4],q[5];
u1(0.08564916714362436) q[5];
cx q[4],q[5];
cx q[4],q[8];
u1(0.08564916714362436) q[8];
cx q[4],q[8];
cx q[5],q[7];
u1(0.08564916714362436) q[7];
cx q[5],q[7];
cx q[5],q[9];
u1(0.08564916714362436) q[9];
cx q[5],q[9];
cx q[6],q[7];
u1(0.08564916714362436) q[7];
cx q[6],q[7];
cx q[6],q[9];
u1(0.08564916714362436) q[9];
cx q[6],q[9];
cx q[7],q[8];
u1(0.08564916714362436) q[8];
cx q[7],q[8];
h q[0];
rz(0.4736210131921994) q[0];
h q[0];
h q[1];
rz(0.4736210131921994) q[1];
h q[1];
h q[2];
rz(0.4736210131921994) q[2];
h q[2];
h q[3];
rz(0.4736210131921994) q[3];
h q[3];
h q[4];
rz(0.4736210131921994) q[4];
h q[4];
h q[5];
rz(0.4736210131921994) q[5];
h q[5];
h q[6];
rz(0.4736210131921994) q[6];
h q[6];
h q[7];
rz(0.4736210131921994) q[7];
h q[7];
h q[8];
rz(0.4736210131921994) q[8];
h q[8];
h q[9];
rz(0.4736210131921994) q[9];
h q[9];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];