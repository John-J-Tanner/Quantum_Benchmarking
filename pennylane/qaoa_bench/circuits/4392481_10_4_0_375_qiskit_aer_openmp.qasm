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
cx q[0],q[3];
u1(0.5381643514719432) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.5381643514719432) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.5381643514719432) q[7];
cx q[0],q[7];
cx q[1],q[4];
u1(0.5381643514719432) q[4];
cx q[1],q[4];
cx q[1],q[5];
u1(0.5381643514719432) q[5];
cx q[1],q[5];
cx q[1],q[6];
u1(0.5381643514719432) q[6];
cx q[1],q[6];
cx q[1],q[9];
u1(0.5381643514719432) q[9];
cx q[1],q[9];
cx q[2],q[8];
u1(0.5381643514719432) q[8];
cx q[2],q[8];
cx q[3],q[4];
u1(0.5381643514719432) q[4];
cx q[3],q[4];
cx q[4],q[5];
u1(0.5381643514719432) q[5];
cx q[4],q[5];
cx q[4],q[7];
u1(0.5381643514719432) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.5381643514719432) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.5381643514719432) q[6];
cx q[5],q[6];
cx q[5],q[7];
u1(0.5381643514719432) q[7];
cx q[5],q[7];
cx q[5],q[8];
u1(0.5381643514719432) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.5381643514719432) q[9];
cx q[5],q[9];
cx q[7],q[8];
u1(0.5381643514719432) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.5381643514719432) q[9];
cx q[7],q[9];
h q[0];
rz(1.9748899803729332) q[0];
h q[0];
h q[1];
rz(1.9748899803729332) q[1];
h q[1];
h q[2];
rz(1.9748899803729332) q[2];
h q[2];
h q[3];
rz(1.9748899803729332) q[3];
h q[3];
h q[4];
rz(1.9748899803729332) q[4];
h q[4];
h q[5];
rz(1.9748899803729332) q[5];
h q[5];
h q[6];
rz(1.9748899803729332) q[6];
h q[6];
h q[7];
rz(1.9748899803729332) q[7];
h q[7];
h q[8];
rz(1.9748899803729332) q[8];
h q[8];
h q[9];
rz(1.9748899803729332) q[9];
h q[9];
cx q[0],q[3];
u1(0.34327086981333843) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.34327086981333843) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.34327086981333843) q[7];
cx q[0],q[7];
cx q[1],q[4];
u1(0.34327086981333843) q[4];
cx q[1],q[4];
cx q[1],q[5];
u1(0.34327086981333843) q[5];
cx q[1],q[5];
cx q[1],q[6];
u1(0.34327086981333843) q[6];
cx q[1],q[6];
cx q[1],q[9];
u1(0.34327086981333843) q[9];
cx q[1],q[9];
cx q[2],q[8];
u1(0.34327086981333843) q[8];
cx q[2],q[8];
cx q[3],q[4];
u1(0.34327086981333843) q[4];
cx q[3],q[4];
cx q[4],q[5];
u1(0.34327086981333843) q[5];
cx q[4],q[5];
cx q[4],q[7];
u1(0.34327086981333843) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.34327086981333843) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.34327086981333843) q[6];
cx q[5],q[6];
cx q[5],q[7];
u1(0.34327086981333843) q[7];
cx q[5],q[7];
cx q[5],q[8];
u1(0.34327086981333843) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.34327086981333843) q[9];
cx q[5],q[9];
cx q[7],q[8];
u1(0.34327086981333843) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.34327086981333843) q[9];
cx q[7],q[9];
h q[0];
rz(1.2655125452142921) q[0];
h q[0];
h q[1];
rz(1.2655125452142921) q[1];
h q[1];
h q[2];
rz(1.2655125452142921) q[2];
h q[2];
h q[3];
rz(1.2655125452142921) q[3];
h q[3];
h q[4];
rz(1.2655125452142921) q[4];
h q[4];
h q[5];
rz(1.2655125452142921) q[5];
h q[5];
h q[6];
rz(1.2655125452142921) q[6];
h q[6];
h q[7];
rz(1.2655125452142921) q[7];
h q[7];
h q[8];
rz(1.2655125452142921) q[8];
h q[8];
h q[9];
rz(1.2655125452142921) q[9];
h q[9];
cx q[0],q[3];
u1(0.36906723979537825) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.36906723979537825) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.36906723979537825) q[7];
cx q[0],q[7];
cx q[1],q[4];
u1(0.36906723979537825) q[4];
cx q[1],q[4];
cx q[1],q[5];
u1(0.36906723979537825) q[5];
cx q[1],q[5];
cx q[1],q[6];
u1(0.36906723979537825) q[6];
cx q[1],q[6];
cx q[1],q[9];
u1(0.36906723979537825) q[9];
cx q[1],q[9];
cx q[2],q[8];
u1(0.36906723979537825) q[8];
cx q[2],q[8];
cx q[3],q[4];
u1(0.36906723979537825) q[4];
cx q[3],q[4];
cx q[4],q[5];
u1(0.36906723979537825) q[5];
cx q[4],q[5];
cx q[4],q[7];
u1(0.36906723979537825) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.36906723979537825) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.36906723979537825) q[6];
cx q[5],q[6];
cx q[5],q[7];
u1(0.36906723979537825) q[7];
cx q[5],q[7];
cx q[5],q[8];
u1(0.36906723979537825) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.36906723979537825) q[9];
cx q[5],q[9];
cx q[7],q[8];
u1(0.36906723979537825) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.36906723979537825) q[9];
cx q[7],q[9];
h q[0];
rz(1.3486478610089871) q[0];
h q[0];
h q[1];
rz(1.3486478610089871) q[1];
h q[1];
h q[2];
rz(1.3486478610089871) q[2];
h q[2];
h q[3];
rz(1.3486478610089871) q[3];
h q[3];
h q[4];
rz(1.3486478610089871) q[4];
h q[4];
h q[5];
rz(1.3486478610089871) q[5];
h q[5];
h q[6];
rz(1.3486478610089871) q[6];
h q[6];
h q[7];
rz(1.3486478610089871) q[7];
h q[7];
h q[8];
rz(1.3486478610089871) q[8];
h q[8];
h q[9];
rz(1.3486478610089871) q[9];
h q[9];
cx q[0],q[3];
u1(0.37449676558788236) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.37449676558788236) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.37449676558788236) q[7];
cx q[0],q[7];
cx q[1],q[4];
u1(0.37449676558788236) q[4];
cx q[1],q[4];
cx q[1],q[5];
u1(0.37449676558788236) q[5];
cx q[1],q[5];
cx q[1],q[6];
u1(0.37449676558788236) q[6];
cx q[1],q[6];
cx q[1],q[9];
u1(0.37449676558788236) q[9];
cx q[1],q[9];
cx q[2],q[8];
u1(0.37449676558788236) q[8];
cx q[2],q[8];
cx q[3],q[4];
u1(0.37449676558788236) q[4];
cx q[3],q[4];
cx q[4],q[5];
u1(0.37449676558788236) q[5];
cx q[4],q[5];
cx q[4],q[7];
u1(0.37449676558788236) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.37449676558788236) q[9];
cx q[4],q[9];
cx q[5],q[6];
u1(0.37449676558788236) q[6];
cx q[5],q[6];
cx q[5],q[7];
u1(0.37449676558788236) q[7];
cx q[5],q[7];
cx q[5],q[8];
u1(0.37449676558788236) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.37449676558788236) q[9];
cx q[5],q[9];
cx q[7],q[8];
u1(0.37449676558788236) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.37449676558788236) q[9];
cx q[7],q[9];
h q[0];
rz(0.6599269107709167) q[0];
h q[0];
h q[1];
rz(0.6599269107709167) q[1];
h q[1];
h q[2];
rz(0.6599269107709167) q[2];
h q[2];
h q[3];
rz(0.6599269107709167) q[3];
h q[3];
h q[4];
rz(0.6599269107709167) q[4];
h q[4];
h q[5];
rz(0.6599269107709167) q[5];
h q[5];
h q[6];
rz(0.6599269107709167) q[6];
h q[6];
h q[7];
rz(0.6599269107709167) q[7];
h q[7];
h q[8];
rz(0.6599269107709167) q[8];
h q[8];
h q[9];
rz(0.6599269107709167) q[9];
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
