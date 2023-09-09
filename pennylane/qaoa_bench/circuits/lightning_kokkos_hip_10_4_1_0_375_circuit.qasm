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
u1(0.5791410932265623) q[1];
cx q[0],q[1];
cx q[0],q[5];
u1(0.5791410932265623) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.5791410932265623) q[7];
cx q[0],q[7];
cx q[1],q[2];
u1(0.5791410932265623) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.5791410932265623) q[4];
cx q[1],q[4];
cx q[2],q[4];
u1(0.5791410932265623) q[4];
cx q[2],q[4];
cx q[2],q[7];
u1(0.5791410932265623) q[7];
cx q[2],q[7];
cx q[3],q[5];
u1(0.5791410932265623) q[5];
cx q[3],q[5];
cx q[3],q[7];
u1(0.5791410932265623) q[7];
cx q[3],q[7];
cx q[3],q[8];
u1(0.5791410932265623) q[8];
cx q[3],q[8];
cx q[4],q[5];
u1(0.5791410932265623) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.5791410932265623) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.5791410932265623) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.5791410932265623) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.5791410932265623) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.5791410932265623) q[8];
cx q[5],q[8];
cx q[6],q[7];
u1(0.5791410932265623) q[7];
cx q[6],q[7];
cx q[6],q[8];
u1(0.5791410932265623) q[8];
cx q[6],q[8];
cx q[7],q[9];
u1(0.5791410932265623) q[9];
cx q[7],q[9];
cx q[8],q[9];
u1(0.5791410932265623) q[9];
cx q[8],q[9];
h q[0];
rz(1.1268130734060484) q[0];
h q[0];
h q[1];
rz(1.1268130734060484) q[1];
h q[1];
h q[2];
rz(1.1268130734060484) q[2];
h q[2];
h q[3];
rz(1.1268130734060484) q[3];
h q[3];
h q[4];
rz(1.1268130734060484) q[4];
h q[4];
h q[5];
rz(1.1268130734060484) q[5];
h q[5];
h q[6];
rz(1.1268130734060484) q[6];
h q[6];
h q[7];
rz(1.1268130734060484) q[7];
h q[7];
h q[8];
rz(1.1268130734060484) q[8];
h q[8];
h q[9];
rz(1.1268130734060484) q[9];
h q[9];
cx q[0],q[1];
u1(0.5272671894974809) q[1];
cx q[0],q[1];
cx q[0],q[5];
u1(0.5272671894974809) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.5272671894974809) q[7];
cx q[0],q[7];
cx q[1],q[2];
u1(0.5272671894974809) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.5272671894974809) q[4];
cx q[1],q[4];
cx q[2],q[4];
u1(0.5272671894974809) q[4];
cx q[2],q[4];
cx q[2],q[7];
u1(0.5272671894974809) q[7];
cx q[2],q[7];
cx q[3],q[5];
u1(0.5272671894974809) q[5];
cx q[3],q[5];
cx q[3],q[7];
u1(0.5272671894974809) q[7];
cx q[3],q[7];
cx q[3],q[8];
u1(0.5272671894974809) q[8];
cx q[3],q[8];
cx q[4],q[5];
u1(0.5272671894974809) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.5272671894974809) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.5272671894974809) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.5272671894974809) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.5272671894974809) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.5272671894974809) q[8];
cx q[5],q[8];
cx q[6],q[7];
u1(0.5272671894974809) q[7];
cx q[6],q[7];
cx q[6],q[8];
u1(0.5272671894974809) q[8];
cx q[6],q[8];
cx q[7],q[9];
u1(0.5272671894974809) q[9];
cx q[7],q[9];
cx q[8],q[9];
u1(0.5272671894974809) q[9];
cx q[8],q[9];
h q[0];
rz(0.24184011583523568) q[0];
h q[0];
h q[1];
rz(0.24184011583523568) q[1];
h q[1];
h q[2];
rz(0.24184011583523568) q[2];
h q[2];
h q[3];
rz(0.24184011583523568) q[3];
h q[3];
h q[4];
rz(0.24184011583523568) q[4];
h q[4];
h q[5];
rz(0.24184011583523568) q[5];
h q[5];
h q[6];
rz(0.24184011583523568) q[6];
h q[6];
h q[7];
rz(0.24184011583523568) q[7];
h q[7];
h q[8];
rz(0.24184011583523568) q[8];
h q[8];
h q[9];
rz(0.24184011583523568) q[9];
h q[9];
cx q[0],q[1];
u1(0.8201026727189699) q[1];
cx q[0],q[1];
cx q[0],q[5];
u1(0.8201026727189699) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.8201026727189699) q[7];
cx q[0],q[7];
cx q[1],q[2];
u1(0.8201026727189699) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.8201026727189699) q[4];
cx q[1],q[4];
cx q[2],q[4];
u1(0.8201026727189699) q[4];
cx q[2],q[4];
cx q[2],q[7];
u1(0.8201026727189699) q[7];
cx q[2],q[7];
cx q[3],q[5];
u1(0.8201026727189699) q[5];
cx q[3],q[5];
cx q[3],q[7];
u1(0.8201026727189699) q[7];
cx q[3],q[7];
cx q[3],q[8];
u1(0.8201026727189699) q[8];
cx q[3],q[8];
cx q[4],q[5];
u1(0.8201026727189699) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.8201026727189699) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.8201026727189699) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.8201026727189699) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.8201026727189699) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.8201026727189699) q[8];
cx q[5],q[8];
cx q[6],q[7];
u1(0.8201026727189699) q[7];
cx q[6],q[7];
cx q[6],q[8];
u1(0.8201026727189699) q[8];
cx q[6],q[8];
cx q[7],q[9];
u1(0.8201026727189699) q[9];
cx q[7],q[9];
cx q[8],q[9];
u1(0.8201026727189699) q[9];
cx q[8],q[9];
h q[0];
rz(1.4725132529768756) q[0];
h q[0];
h q[1];
rz(1.4725132529768756) q[1];
h q[1];
h q[2];
rz(1.4725132529768756) q[2];
h q[2];
h q[3];
rz(1.4725132529768756) q[3];
h q[3];
h q[4];
rz(1.4725132529768756) q[4];
h q[4];
h q[5];
rz(1.4725132529768756) q[5];
h q[5];
h q[6];
rz(1.4725132529768756) q[6];
h q[6];
h q[7];
rz(1.4725132529768756) q[7];
h q[7];
h q[8];
rz(1.4725132529768756) q[8];
h q[8];
h q[9];
rz(1.4725132529768756) q[9];
h q[9];
cx q[0],q[1];
u1(0.7446141011197945) q[1];
cx q[0],q[1];
cx q[0],q[5];
u1(0.7446141011197945) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.7446141011197945) q[7];
cx q[0],q[7];
cx q[1],q[2];
u1(0.7446141011197945) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.7446141011197945) q[4];
cx q[1],q[4];
cx q[2],q[4];
u1(0.7446141011197945) q[4];
cx q[2],q[4];
cx q[2],q[7];
u1(0.7446141011197945) q[7];
cx q[2],q[7];
cx q[3],q[5];
u1(0.7446141011197945) q[5];
cx q[3],q[5];
cx q[3],q[7];
u1(0.7446141011197945) q[7];
cx q[3],q[7];
cx q[3],q[8];
u1(0.7446141011197945) q[8];
cx q[3],q[8];
cx q[4],q[5];
u1(0.7446141011197945) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.7446141011197945) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.7446141011197945) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.7446141011197945) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.7446141011197945) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.7446141011197945) q[8];
cx q[5],q[8];
cx q[6],q[7];
u1(0.7446141011197945) q[7];
cx q[6],q[7];
cx q[6],q[8];
u1(0.7446141011197945) q[8];
cx q[6],q[8];
cx q[7],q[9];
u1(0.7446141011197945) q[9];
cx q[7],q[9];
cx q[8],q[9];
u1(0.7446141011197945) q[9];
cx q[8],q[9];
h q[0];
rz(1.6976147513399156) q[0];
h q[0];
h q[1];
rz(1.6976147513399156) q[1];
h q[1];
h q[2];
rz(1.6976147513399156) q[2];
h q[2];
h q[3];
rz(1.6976147513399156) q[3];
h q[3];
h q[4];
rz(1.6976147513399156) q[4];
h q[4];
h q[5];
rz(1.6976147513399156) q[5];
h q[5];
h q[6];
rz(1.6976147513399156) q[6];
h q[6];
h q[7];
rz(1.6976147513399156) q[7];
h q[7];
h q[8];
rz(1.6976147513399156) q[8];
h q[8];
h q[9];
rz(1.6976147513399156) q[9];
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