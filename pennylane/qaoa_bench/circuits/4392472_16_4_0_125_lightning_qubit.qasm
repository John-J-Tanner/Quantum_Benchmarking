OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
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
cx q[0],q[11];
u1(0.2800759626301593) q[11];
cx q[0],q[11];
cx q[0],q[12];
u1(0.2800759626301593) q[12];
cx q[0],q[12];
cx q[1],q[9];
u1(0.2800759626301593) q[9];
cx q[1],q[9];
cx q[2],q[6];
u1(0.2800759626301593) q[6];
cx q[2],q[6];
cx q[2],q[15];
u1(0.2800759626301593) q[15];
cx q[2],q[15];
cx q[3],q[11];
u1(0.2800759626301593) q[11];
cx q[3],q[11];
cx q[4],q[15];
u1(0.2800759626301593) q[15];
cx q[4],q[15];
cx q[5],q[7];
u1(0.2800759626301593) q[7];
cx q[5],q[7];
cx q[6],q[13];
u1(0.2800759626301593) q[13];
cx q[6],q[13];
cx q[8],q[15];
u1(0.2800759626301593) q[15];
cx q[8],q[15];
cx q[9],q[14];
u1(0.2800759626301593) q[14];
cx q[9],q[14];
h q[0];
rz(0.8183368175484098) q[0];
h q[0];
h q[1];
rz(0.8183368175484098) q[1];
h q[1];
h q[2];
rz(0.8183368175484098) q[2];
h q[2];
h q[3];
rz(0.8183368175484098) q[3];
h q[3];
h q[4];
rz(0.8183368175484098) q[4];
h q[4];
h q[5];
rz(0.8183368175484098) q[5];
h q[5];
h q[6];
rz(0.8183368175484098) q[6];
h q[6];
h q[7];
rz(0.8183368175484098) q[7];
h q[7];
h q[8];
rz(0.8183368175484098) q[8];
h q[8];
h q[9];
rz(0.8183368175484098) q[9];
h q[9];
h q[10];
rz(0.8183368175484098) q[10];
h q[10];
h q[11];
rz(0.8183368175484098) q[11];
h q[11];
h q[12];
rz(0.8183368175484098) q[12];
h q[12];
h q[13];
rz(0.8183368175484098) q[13];
h q[13];
h q[14];
rz(0.8183368175484098) q[14];
h q[14];
h q[15];
rz(0.8183368175484098) q[15];
h q[15];
cx q[0],q[11];
u1(0.46114670980294215) q[11];
cx q[0],q[11];
cx q[0],q[12];
u1(0.46114670980294215) q[12];
cx q[0],q[12];
cx q[1],q[9];
u1(0.46114670980294215) q[9];
cx q[1],q[9];
cx q[2],q[6];
u1(0.46114670980294215) q[6];
cx q[2],q[6];
cx q[2],q[15];
u1(0.46114670980294215) q[15];
cx q[2],q[15];
cx q[3],q[11];
u1(0.46114670980294215) q[11];
cx q[3],q[11];
cx q[4],q[15];
u1(0.46114670980294215) q[15];
cx q[4],q[15];
cx q[5],q[7];
u1(0.46114670980294215) q[7];
cx q[5],q[7];
cx q[6],q[13];
u1(0.46114670980294215) q[13];
cx q[6],q[13];
cx q[8],q[15];
u1(0.46114670980294215) q[15];
cx q[8],q[15];
cx q[9],q[14];
u1(0.46114670980294215) q[14];
cx q[9],q[14];
h q[0];
rz(0.14327909440932984) q[0];
h q[0];
h q[1];
rz(0.14327909440932984) q[1];
h q[1];
h q[2];
rz(0.14327909440932984) q[2];
h q[2];
h q[3];
rz(0.14327909440932984) q[3];
h q[3];
h q[4];
rz(0.14327909440932984) q[4];
h q[4];
h q[5];
rz(0.14327909440932984) q[5];
h q[5];
h q[6];
rz(0.14327909440932984) q[6];
h q[6];
h q[7];
rz(0.14327909440932984) q[7];
h q[7];
h q[8];
rz(0.14327909440932984) q[8];
h q[8];
h q[9];
rz(0.14327909440932984) q[9];
h q[9];
h q[10];
rz(0.14327909440932984) q[10];
h q[10];
h q[11];
rz(0.14327909440932984) q[11];
h q[11];
h q[12];
rz(0.14327909440932984) q[12];
h q[12];
h q[13];
rz(0.14327909440932984) q[13];
h q[13];
h q[14];
rz(0.14327909440932984) q[14];
h q[14];
h q[15];
rz(0.14327909440932984) q[15];
h q[15];
cx q[0],q[11];
u1(0.12171969257644188) q[11];
cx q[0],q[11];
cx q[0],q[12];
u1(0.12171969257644188) q[12];
cx q[0],q[12];
cx q[1],q[9];
u1(0.12171969257644188) q[9];
cx q[1],q[9];
cx q[2],q[6];
u1(0.12171969257644188) q[6];
cx q[2],q[6];
cx q[2],q[15];
u1(0.12171969257644188) q[15];
cx q[2],q[15];
cx q[3],q[11];
u1(0.12171969257644188) q[11];
cx q[3],q[11];
cx q[4],q[15];
u1(0.12171969257644188) q[15];
cx q[4],q[15];
cx q[5],q[7];
u1(0.12171969257644188) q[7];
cx q[5],q[7];
cx q[6],q[13];
u1(0.12171969257644188) q[13];
cx q[6],q[13];
cx q[8],q[15];
u1(0.12171969257644188) q[15];
cx q[8],q[15];
cx q[9],q[14];
u1(0.12171969257644188) q[14];
cx q[9],q[14];
h q[0];
rz(0.1979333516370636) q[0];
h q[0];
h q[1];
rz(0.1979333516370636) q[1];
h q[1];
h q[2];
rz(0.1979333516370636) q[2];
h q[2];
h q[3];
rz(0.1979333516370636) q[3];
h q[3];
h q[4];
rz(0.1979333516370636) q[4];
h q[4];
h q[5];
rz(0.1979333516370636) q[5];
h q[5];
h q[6];
rz(0.1979333516370636) q[6];
h q[6];
h q[7];
rz(0.1979333516370636) q[7];
h q[7];
h q[8];
rz(0.1979333516370636) q[8];
h q[8];
h q[9];
rz(0.1979333516370636) q[9];
h q[9];
h q[10];
rz(0.1979333516370636) q[10];
h q[10];
h q[11];
rz(0.1979333516370636) q[11];
h q[11];
h q[12];
rz(0.1979333516370636) q[12];
h q[12];
h q[13];
rz(0.1979333516370636) q[13];
h q[13];
h q[14];
rz(0.1979333516370636) q[14];
h q[14];
h q[15];
rz(0.1979333516370636) q[15];
h q[15];
cx q[0],q[11];
u1(0.5226083517189325) q[11];
cx q[0],q[11];
cx q[0],q[12];
u1(0.5226083517189325) q[12];
cx q[0],q[12];
cx q[1],q[9];
u1(0.5226083517189325) q[9];
cx q[1],q[9];
cx q[2],q[6];
u1(0.5226083517189325) q[6];
cx q[2],q[6];
cx q[2],q[15];
u1(0.5226083517189325) q[15];
cx q[2],q[15];
cx q[3],q[11];
u1(0.5226083517189325) q[11];
cx q[3],q[11];
cx q[4],q[15];
u1(0.5226083517189325) q[15];
cx q[4],q[15];
cx q[5],q[7];
u1(0.5226083517189325) q[7];
cx q[5],q[7];
cx q[6],q[13];
u1(0.5226083517189325) q[13];
cx q[6],q[13];
cx q[8],q[15];
u1(0.5226083517189325) q[15];
cx q[8],q[15];
cx q[9],q[14];
u1(0.5226083517189325) q[14];
cx q[9],q[14];
h q[0];
rz(1.9725767279334938) q[0];
h q[0];
h q[1];
rz(1.9725767279334938) q[1];
h q[1];
h q[2];
rz(1.9725767279334938) q[2];
h q[2];
h q[3];
rz(1.9725767279334938) q[3];
h q[3];
h q[4];
rz(1.9725767279334938) q[4];
h q[4];
h q[5];
rz(1.9725767279334938) q[5];
h q[5];
h q[6];
rz(1.9725767279334938) q[6];
h q[6];
h q[7];
rz(1.9725767279334938) q[7];
h q[7];
h q[8];
rz(1.9725767279334938) q[8];
h q[8];
h q[9];
rz(1.9725767279334938) q[9];
h q[9];
h q[10];
rz(1.9725767279334938) q[10];
h q[10];
h q[11];
rz(1.9725767279334938) q[11];
h q[11];
h q[12];
rz(1.9725767279334938) q[12];
h q[12];
h q[13];
rz(1.9725767279334938) q[13];
h q[13];
h q[14];
rz(1.9725767279334938) q[14];
h q[14];
h q[15];
rz(1.9725767279334938) q[15];
h q[15];
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
