OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
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
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
cx q[0],q[15];
u1(0.18063309122416948) q[15];
cx q[0],q[15];
cx q[1],q[2];
u1(0.18063309122416948) q[2];
cx q[1],q[2];
cx q[2],q[11];
u1(0.18063309122416948) q[11];
cx q[2],q[11];
cx q[2],q[17];
u1(0.18063309122416948) q[17];
cx q[2],q[17];
cx q[5],q[8];
u1(0.18063309122416948) q[8];
cx q[5],q[8];
cx q[6],q[16];
u1(0.18063309122416948) q[16];
cx q[6],q[16];
cx q[8],q[15];
u1(0.18063309122416948) q[15];
cx q[8],q[15];
cx q[8],q[17];
u1(0.18063309122416948) q[17];
cx q[8],q[17];
cx q[10],q[12];
u1(0.18063309122416948) q[12];
cx q[10],q[12];
cx q[10],q[13];
u1(0.18063309122416948) q[13];
cx q[10],q[13];
cx q[11],q[15];
u1(0.18063309122416948) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.18063309122416948) q[16];
cx q[11],q[16];
cx q[12],q[18];
u1(0.18063309122416948) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.18063309122416948) q[14];
cx q[13],q[14];
cx q[15],q[17];
u1(0.18063309122416948) q[17];
cx q[15],q[17];
h q[0];
rz(1.3660092460559432) q[0];
h q[0];
h q[1];
rz(1.3660092460559432) q[1];
h q[1];
h q[2];
rz(1.3660092460559432) q[2];
h q[2];
h q[3];
rz(1.3660092460559432) q[3];
h q[3];
h q[4];
rz(1.3660092460559432) q[4];
h q[4];
h q[5];
rz(1.3660092460559432) q[5];
h q[5];
h q[6];
rz(1.3660092460559432) q[6];
h q[6];
h q[7];
rz(1.3660092460559432) q[7];
h q[7];
h q[8];
rz(1.3660092460559432) q[8];
h q[8];
h q[9];
rz(1.3660092460559432) q[9];
h q[9];
h q[10];
rz(1.3660092460559432) q[10];
h q[10];
h q[11];
rz(1.3660092460559432) q[11];
h q[11];
h q[12];
rz(1.3660092460559432) q[12];
h q[12];
h q[13];
rz(1.3660092460559432) q[13];
h q[13];
h q[14];
rz(1.3660092460559432) q[14];
h q[14];
h q[15];
rz(1.3660092460559432) q[15];
h q[15];
h q[16];
rz(1.3660092460559432) q[16];
h q[16];
h q[17];
rz(1.3660092460559432) q[17];
h q[17];
h q[18];
rz(1.3660092460559432) q[18];
h q[18];
h q[19];
rz(1.3660092460559432) q[19];
h q[19];
cx q[0],q[15];
u1(0.3982367607356684) q[15];
cx q[0],q[15];
cx q[1],q[2];
u1(0.3982367607356684) q[2];
cx q[1],q[2];
cx q[2],q[11];
u1(0.3982367607356684) q[11];
cx q[2],q[11];
cx q[2],q[17];
u1(0.3982367607356684) q[17];
cx q[2],q[17];
cx q[5],q[8];
u1(0.3982367607356684) q[8];
cx q[5],q[8];
cx q[6],q[16];
u1(0.3982367607356684) q[16];
cx q[6],q[16];
cx q[8],q[15];
u1(0.3982367607356684) q[15];
cx q[8],q[15];
cx q[8],q[17];
u1(0.3982367607356684) q[17];
cx q[8],q[17];
cx q[10],q[12];
u1(0.3982367607356684) q[12];
cx q[10],q[12];
cx q[10],q[13];
u1(0.3982367607356684) q[13];
cx q[10],q[13];
cx q[11],q[15];
u1(0.3982367607356684) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.3982367607356684) q[16];
cx q[11],q[16];
cx q[12],q[18];
u1(0.3982367607356684) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.3982367607356684) q[14];
cx q[13],q[14];
cx q[15],q[17];
u1(0.3982367607356684) q[17];
cx q[15],q[17];
h q[0];
rz(1.0221618304271893) q[0];
h q[0];
h q[1];
rz(1.0221618304271893) q[1];
h q[1];
h q[2];
rz(1.0221618304271893) q[2];
h q[2];
h q[3];
rz(1.0221618304271893) q[3];
h q[3];
h q[4];
rz(1.0221618304271893) q[4];
h q[4];
h q[5];
rz(1.0221618304271893) q[5];
h q[5];
h q[6];
rz(1.0221618304271893) q[6];
h q[6];
h q[7];
rz(1.0221618304271893) q[7];
h q[7];
h q[8];
rz(1.0221618304271893) q[8];
h q[8];
h q[9];
rz(1.0221618304271893) q[9];
h q[9];
h q[10];
rz(1.0221618304271893) q[10];
h q[10];
h q[11];
rz(1.0221618304271893) q[11];
h q[11];
h q[12];
rz(1.0221618304271893) q[12];
h q[12];
h q[13];
rz(1.0221618304271893) q[13];
h q[13];
h q[14];
rz(1.0221618304271893) q[14];
h q[14];
h q[15];
rz(1.0221618304271893) q[15];
h q[15];
h q[16];
rz(1.0221618304271893) q[16];
h q[16];
h q[17];
rz(1.0221618304271893) q[17];
h q[17];
h q[18];
rz(1.0221618304271893) q[18];
h q[18];
h q[19];
rz(1.0221618304271893) q[19];
h q[19];
cx q[0],q[15];
u1(0.8938324035698779) q[15];
cx q[0],q[15];
cx q[1],q[2];
u1(0.8938324035698779) q[2];
cx q[1],q[2];
cx q[2],q[11];
u1(0.8938324035698779) q[11];
cx q[2],q[11];
cx q[2],q[17];
u1(0.8938324035698779) q[17];
cx q[2],q[17];
cx q[5],q[8];
u1(0.8938324035698779) q[8];
cx q[5],q[8];
cx q[6],q[16];
u1(0.8938324035698779) q[16];
cx q[6],q[16];
cx q[8],q[15];
u1(0.8938324035698779) q[15];
cx q[8],q[15];
cx q[8],q[17];
u1(0.8938324035698779) q[17];
cx q[8],q[17];
cx q[10],q[12];
u1(0.8938324035698779) q[12];
cx q[10],q[12];
cx q[10],q[13];
u1(0.8938324035698779) q[13];
cx q[10],q[13];
cx q[11],q[15];
u1(0.8938324035698779) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.8938324035698779) q[16];
cx q[11],q[16];
cx q[12],q[18];
u1(0.8938324035698779) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.8938324035698779) q[14];
cx q[13],q[14];
cx q[15],q[17];
u1(0.8938324035698779) q[17];
cx q[15],q[17];
h q[0];
rz(1.07085310616461) q[0];
h q[0];
h q[1];
rz(1.07085310616461) q[1];
h q[1];
h q[2];
rz(1.07085310616461) q[2];
h q[2];
h q[3];
rz(1.07085310616461) q[3];
h q[3];
h q[4];
rz(1.07085310616461) q[4];
h q[4];
h q[5];
rz(1.07085310616461) q[5];
h q[5];
h q[6];
rz(1.07085310616461) q[6];
h q[6];
h q[7];
rz(1.07085310616461) q[7];
h q[7];
h q[8];
rz(1.07085310616461) q[8];
h q[8];
h q[9];
rz(1.07085310616461) q[9];
h q[9];
h q[10];
rz(1.07085310616461) q[10];
h q[10];
h q[11];
rz(1.07085310616461) q[11];
h q[11];
h q[12];
rz(1.07085310616461) q[12];
h q[12];
h q[13];
rz(1.07085310616461) q[13];
h q[13];
h q[14];
rz(1.07085310616461) q[14];
h q[14];
h q[15];
rz(1.07085310616461) q[15];
h q[15];
h q[16];
rz(1.07085310616461) q[16];
h q[16];
h q[17];
rz(1.07085310616461) q[17];
h q[17];
h q[18];
rz(1.07085310616461) q[18];
h q[18];
h q[19];
rz(1.07085310616461) q[19];
h q[19];
cx q[0],q[15];
u1(0.4033860075146668) q[15];
cx q[0],q[15];
cx q[1],q[2];
u1(0.4033860075146668) q[2];
cx q[1],q[2];
cx q[2],q[11];
u1(0.4033860075146668) q[11];
cx q[2],q[11];
cx q[2],q[17];
u1(0.4033860075146668) q[17];
cx q[2],q[17];
cx q[5],q[8];
u1(0.4033860075146668) q[8];
cx q[5],q[8];
cx q[6],q[16];
u1(0.4033860075146668) q[16];
cx q[6],q[16];
cx q[8],q[15];
u1(0.4033860075146668) q[15];
cx q[8],q[15];
cx q[8],q[17];
u1(0.4033860075146668) q[17];
cx q[8],q[17];
cx q[10],q[12];
u1(0.4033860075146668) q[12];
cx q[10],q[12];
cx q[10],q[13];
u1(0.4033860075146668) q[13];
cx q[10],q[13];
cx q[11],q[15];
u1(0.4033860075146668) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.4033860075146668) q[16];
cx q[11],q[16];
cx q[12],q[18];
u1(0.4033860075146668) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.4033860075146668) q[14];
cx q[13],q[14];
cx q[15],q[17];
u1(0.4033860075146668) q[17];
cx q[15],q[17];
h q[0];
rz(1.7507089136405813) q[0];
h q[0];
h q[1];
rz(1.7507089136405813) q[1];
h q[1];
h q[2];
rz(1.7507089136405813) q[2];
h q[2];
h q[3];
rz(1.7507089136405813) q[3];
h q[3];
h q[4];
rz(1.7507089136405813) q[4];
h q[4];
h q[5];
rz(1.7507089136405813) q[5];
h q[5];
h q[6];
rz(1.7507089136405813) q[6];
h q[6];
h q[7];
rz(1.7507089136405813) q[7];
h q[7];
h q[8];
rz(1.7507089136405813) q[8];
h q[8];
h q[9];
rz(1.7507089136405813) q[9];
h q[9];
h q[10];
rz(1.7507089136405813) q[10];
h q[10];
h q[11];
rz(1.7507089136405813) q[11];
h q[11];
h q[12];
rz(1.7507089136405813) q[12];
h q[12];
h q[13];
rz(1.7507089136405813) q[13];
h q[13];
h q[14];
rz(1.7507089136405813) q[14];
h q[14];
h q[15];
rz(1.7507089136405813) q[15];
h q[15];
h q[16];
rz(1.7507089136405813) q[16];
h q[16];
h q[17];
rz(1.7507089136405813) q[17];
h q[17];
h q[18];
rz(1.7507089136405813) q[18];
h q[18];
h q[19];
rz(1.7507089136405813) q[19];
h q[19];
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
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];