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
cx q[0],q[9];
u1(0.9498007555454135) q[9];
cx q[0],q[9];
cx q[0],q[16];
u1(0.9498007555454135) q[16];
cx q[0],q[16];
cx q[2],q[11];
u1(0.9498007555454135) q[11];
cx q[2],q[11];
cx q[2],q[15];
u1(0.9498007555454135) q[15];
cx q[2],q[15];
cx q[3],q[17];
u1(0.9498007555454135) q[17];
cx q[3],q[17];
cx q[4],q[9];
u1(0.9498007555454135) q[9];
cx q[4],q[9];
cx q[4],q[13];
u1(0.9498007555454135) q[13];
cx q[4],q[13];
cx q[4],q[14];
u1(0.9498007555454135) q[14];
cx q[4],q[14];
cx q[5],q[13];
u1(0.9498007555454135) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.9498007555454135) q[19];
cx q[5],q[19];
cx q[6],q[9];
u1(0.9498007555454135) q[9];
cx q[6],q[9];
cx q[7],q[11];
u1(0.9498007555454135) q[11];
cx q[7],q[11];
cx q[7],q[18];
u1(0.9498007555454135) q[18];
cx q[7],q[18];
cx q[9],q[10];
u1(0.9498007555454135) q[10];
cx q[9],q[10];
cx q[9],q[16];
u1(0.9498007555454135) q[16];
cx q[9],q[16];
cx q[9],q[17];
u1(0.9498007555454135) q[17];
cx q[9],q[17];
cx q[9],q[19];
u1(0.9498007555454135) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.9498007555454135) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.9498007555454135) q[16];
cx q[10],q[16];
cx q[11],q[18];
u1(0.9498007555454135) q[18];
cx q[11],q[18];
cx q[14],q[18];
u1(0.9498007555454135) q[18];
cx q[14],q[18];
cx q[15],q[19];
u1(0.9498007555454135) q[19];
cx q[15],q[19];
h q[0];
rz(0.8210032853515192) q[0];
h q[0];
h q[1];
rz(0.8210032853515192) q[1];
h q[1];
h q[2];
rz(0.8210032853515192) q[2];
h q[2];
h q[3];
rz(0.8210032853515192) q[3];
h q[3];
h q[4];
rz(0.8210032853515192) q[4];
h q[4];
h q[5];
rz(0.8210032853515192) q[5];
h q[5];
h q[6];
rz(0.8210032853515192) q[6];
h q[6];
h q[7];
rz(0.8210032853515192) q[7];
h q[7];
h q[8];
rz(0.8210032853515192) q[8];
h q[8];
h q[9];
rz(0.8210032853515192) q[9];
h q[9];
h q[10];
rz(0.8210032853515192) q[10];
h q[10];
h q[11];
rz(0.8210032853515192) q[11];
h q[11];
h q[12];
rz(0.8210032853515192) q[12];
h q[12];
h q[13];
rz(0.8210032853515192) q[13];
h q[13];
h q[14];
rz(0.8210032853515192) q[14];
h q[14];
h q[15];
rz(0.8210032853515192) q[15];
h q[15];
h q[16];
rz(0.8210032853515192) q[16];
h q[16];
h q[17];
rz(0.8210032853515192) q[17];
h q[17];
h q[18];
rz(0.8210032853515192) q[18];
h q[18];
h q[19];
rz(0.8210032853515192) q[19];
h q[19];
cx q[0],q[9];
u1(0.33760005100112056) q[9];
cx q[0],q[9];
cx q[0],q[16];
u1(0.33760005100112056) q[16];
cx q[0],q[16];
cx q[2],q[11];
u1(0.33760005100112056) q[11];
cx q[2],q[11];
cx q[2],q[15];
u1(0.33760005100112056) q[15];
cx q[2],q[15];
cx q[3],q[17];
u1(0.33760005100112056) q[17];
cx q[3],q[17];
cx q[4],q[9];
u1(0.33760005100112056) q[9];
cx q[4],q[9];
cx q[4],q[13];
u1(0.33760005100112056) q[13];
cx q[4],q[13];
cx q[4],q[14];
u1(0.33760005100112056) q[14];
cx q[4],q[14];
cx q[5],q[13];
u1(0.33760005100112056) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.33760005100112056) q[19];
cx q[5],q[19];
cx q[6],q[9];
u1(0.33760005100112056) q[9];
cx q[6],q[9];
cx q[7],q[11];
u1(0.33760005100112056) q[11];
cx q[7],q[11];
cx q[7],q[18];
u1(0.33760005100112056) q[18];
cx q[7],q[18];
cx q[9],q[10];
u1(0.33760005100112056) q[10];
cx q[9],q[10];
cx q[9],q[16];
u1(0.33760005100112056) q[16];
cx q[9],q[16];
cx q[9],q[17];
u1(0.33760005100112056) q[17];
cx q[9],q[17];
cx q[9],q[19];
u1(0.33760005100112056) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.33760005100112056) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.33760005100112056) q[16];
cx q[10],q[16];
cx q[11],q[18];
u1(0.33760005100112056) q[18];
cx q[11],q[18];
cx q[14],q[18];
u1(0.33760005100112056) q[18];
cx q[14],q[18];
cx q[15],q[19];
u1(0.33760005100112056) q[19];
cx q[15],q[19];
h q[0];
rz(0.18112447744517346) q[0];
h q[0];
h q[1];
rz(0.18112447744517346) q[1];
h q[1];
h q[2];
rz(0.18112447744517346) q[2];
h q[2];
h q[3];
rz(0.18112447744517346) q[3];
h q[3];
h q[4];
rz(0.18112447744517346) q[4];
h q[4];
h q[5];
rz(0.18112447744517346) q[5];
h q[5];
h q[6];
rz(0.18112447744517346) q[6];
h q[6];
h q[7];
rz(0.18112447744517346) q[7];
h q[7];
h q[8];
rz(0.18112447744517346) q[8];
h q[8];
h q[9];
rz(0.18112447744517346) q[9];
h q[9];
h q[10];
rz(0.18112447744517346) q[10];
h q[10];
h q[11];
rz(0.18112447744517346) q[11];
h q[11];
h q[12];
rz(0.18112447744517346) q[12];
h q[12];
h q[13];
rz(0.18112447744517346) q[13];
h q[13];
h q[14];
rz(0.18112447744517346) q[14];
h q[14];
h q[15];
rz(0.18112447744517346) q[15];
h q[15];
h q[16];
rz(0.18112447744517346) q[16];
h q[16];
h q[17];
rz(0.18112447744517346) q[17];
h q[17];
h q[18];
rz(0.18112447744517346) q[18];
h q[18];
h q[19];
rz(0.18112447744517346) q[19];
h q[19];
cx q[0],q[9];
u1(0.5975770053697868) q[9];
cx q[0],q[9];
cx q[0],q[16];
u1(0.5975770053697868) q[16];
cx q[0],q[16];
cx q[2],q[11];
u1(0.5975770053697868) q[11];
cx q[2],q[11];
cx q[2],q[15];
u1(0.5975770053697868) q[15];
cx q[2],q[15];
cx q[3],q[17];
u1(0.5975770053697868) q[17];
cx q[3],q[17];
cx q[4],q[9];
u1(0.5975770053697868) q[9];
cx q[4],q[9];
cx q[4],q[13];
u1(0.5975770053697868) q[13];
cx q[4],q[13];
cx q[4],q[14];
u1(0.5975770053697868) q[14];
cx q[4],q[14];
cx q[5],q[13];
u1(0.5975770053697868) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.5975770053697868) q[19];
cx q[5],q[19];
cx q[6],q[9];
u1(0.5975770053697868) q[9];
cx q[6],q[9];
cx q[7],q[11];
u1(0.5975770053697868) q[11];
cx q[7],q[11];
cx q[7],q[18];
u1(0.5975770053697868) q[18];
cx q[7],q[18];
cx q[9],q[10];
u1(0.5975770053697868) q[10];
cx q[9],q[10];
cx q[9],q[16];
u1(0.5975770053697868) q[16];
cx q[9],q[16];
cx q[9],q[17];
u1(0.5975770053697868) q[17];
cx q[9],q[17];
cx q[9],q[19];
u1(0.5975770053697868) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.5975770053697868) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.5975770053697868) q[16];
cx q[10],q[16];
cx q[11],q[18];
u1(0.5975770053697868) q[18];
cx q[11],q[18];
cx q[14],q[18];
u1(0.5975770053697868) q[18];
cx q[14],q[18];
cx q[15],q[19];
u1(0.5975770053697868) q[19];
cx q[15],q[19];
h q[0];
rz(1.7203341975518056) q[0];
h q[0];
h q[1];
rz(1.7203341975518056) q[1];
h q[1];
h q[2];
rz(1.7203341975518056) q[2];
h q[2];
h q[3];
rz(1.7203341975518056) q[3];
h q[3];
h q[4];
rz(1.7203341975518056) q[4];
h q[4];
h q[5];
rz(1.7203341975518056) q[5];
h q[5];
h q[6];
rz(1.7203341975518056) q[6];
h q[6];
h q[7];
rz(1.7203341975518056) q[7];
h q[7];
h q[8];
rz(1.7203341975518056) q[8];
h q[8];
h q[9];
rz(1.7203341975518056) q[9];
h q[9];
h q[10];
rz(1.7203341975518056) q[10];
h q[10];
h q[11];
rz(1.7203341975518056) q[11];
h q[11];
h q[12];
rz(1.7203341975518056) q[12];
h q[12];
h q[13];
rz(1.7203341975518056) q[13];
h q[13];
h q[14];
rz(1.7203341975518056) q[14];
h q[14];
h q[15];
rz(1.7203341975518056) q[15];
h q[15];
h q[16];
rz(1.7203341975518056) q[16];
h q[16];
h q[17];
rz(1.7203341975518056) q[17];
h q[17];
h q[18];
rz(1.7203341975518056) q[18];
h q[18];
h q[19];
rz(1.7203341975518056) q[19];
h q[19];
cx q[0],q[9];
u1(0.9937820193057371) q[9];
cx q[0],q[9];
cx q[0],q[16];
u1(0.9937820193057371) q[16];
cx q[0],q[16];
cx q[2],q[11];
u1(0.9937820193057371) q[11];
cx q[2],q[11];
cx q[2],q[15];
u1(0.9937820193057371) q[15];
cx q[2],q[15];
cx q[3],q[17];
u1(0.9937820193057371) q[17];
cx q[3],q[17];
cx q[4],q[9];
u1(0.9937820193057371) q[9];
cx q[4],q[9];
cx q[4],q[13];
u1(0.9937820193057371) q[13];
cx q[4],q[13];
cx q[4],q[14];
u1(0.9937820193057371) q[14];
cx q[4],q[14];
cx q[5],q[13];
u1(0.9937820193057371) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.9937820193057371) q[19];
cx q[5],q[19];
cx q[6],q[9];
u1(0.9937820193057371) q[9];
cx q[6],q[9];
cx q[7],q[11];
u1(0.9937820193057371) q[11];
cx q[7],q[11];
cx q[7],q[18];
u1(0.9937820193057371) q[18];
cx q[7],q[18];
cx q[9],q[10];
u1(0.9937820193057371) q[10];
cx q[9],q[10];
cx q[9],q[16];
u1(0.9937820193057371) q[16];
cx q[9],q[16];
cx q[9],q[17];
u1(0.9937820193057371) q[17];
cx q[9],q[17];
cx q[9],q[19];
u1(0.9937820193057371) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.9937820193057371) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.9937820193057371) q[16];
cx q[10],q[16];
cx q[11],q[18];
u1(0.9937820193057371) q[18];
cx q[11],q[18];
cx q[14],q[18];
u1(0.9937820193057371) q[18];
cx q[14],q[18];
cx q[15],q[19];
u1(0.9937820193057371) q[19];
cx q[15],q[19];
h q[0];
rz(0.45405393163094065) q[0];
h q[0];
h q[1];
rz(0.45405393163094065) q[1];
h q[1];
h q[2];
rz(0.45405393163094065) q[2];
h q[2];
h q[3];
rz(0.45405393163094065) q[3];
h q[3];
h q[4];
rz(0.45405393163094065) q[4];
h q[4];
h q[5];
rz(0.45405393163094065) q[5];
h q[5];
h q[6];
rz(0.45405393163094065) q[6];
h q[6];
h q[7];
rz(0.45405393163094065) q[7];
h q[7];
h q[8];
rz(0.45405393163094065) q[8];
h q[8];
h q[9];
rz(0.45405393163094065) q[9];
h q[9];
h q[10];
rz(0.45405393163094065) q[10];
h q[10];
h q[11];
rz(0.45405393163094065) q[11];
h q[11];
h q[12];
rz(0.45405393163094065) q[12];
h q[12];
h q[13];
rz(0.45405393163094065) q[13];
h q[13];
h q[14];
rz(0.45405393163094065) q[14];
h q[14];
h q[15];
rz(0.45405393163094065) q[15];
h q[15];
h q[16];
rz(0.45405393163094065) q[16];
h q[16];
h q[17];
rz(0.45405393163094065) q[17];
h q[17];
h q[18];
rz(0.45405393163094065) q[18];
h q[18];
h q[19];
rz(0.45405393163094065) q[19];
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