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
cx q[0],q[4];
u1(0.29592239872155834) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.29592239872155834) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.29592239872155834) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.29592239872155834) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.29592239872155834) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.29592239872155834) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.29592239872155834) q[8];
cx q[6],q[8];
h q[0];
rz(0.608723780242546) q[0];
h q[0];
h q[1];
rz(0.608723780242546) q[1];
h q[1];
h q[2];
rz(0.608723780242546) q[2];
h q[2];
h q[3];
rz(0.608723780242546) q[3];
h q[3];
h q[4];
rz(0.608723780242546) q[4];
h q[4];
h q[5];
rz(0.608723780242546) q[5];
h q[5];
h q[6];
rz(0.608723780242546) q[6];
h q[6];
h q[7];
rz(0.608723780242546) q[7];
h q[7];
h q[8];
rz(0.608723780242546) q[8];
h q[8];
h q[9];
rz(0.608723780242546) q[9];
h q[9];
cx q[0],q[4];
u1(0.07671821387927968) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.07671821387927968) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.07671821387927968) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.07671821387927968) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.07671821387927968) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.07671821387927968) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.07671821387927968) q[8];
cx q[6],q[8];
h q[0];
rz(0.28512904368373126) q[0];
h q[0];
h q[1];
rz(0.28512904368373126) q[1];
h q[1];
h q[2];
rz(0.28512904368373126) q[2];
h q[2];
h q[3];
rz(0.28512904368373126) q[3];
h q[3];
h q[4];
rz(0.28512904368373126) q[4];
h q[4];
h q[5];
rz(0.28512904368373126) q[5];
h q[5];
h q[6];
rz(0.28512904368373126) q[6];
h q[6];
h q[7];
rz(0.28512904368373126) q[7];
h q[7];
h q[8];
rz(0.28512904368373126) q[8];
h q[8];
h q[9];
rz(0.28512904368373126) q[9];
h q[9];
cx q[0],q[4];
u1(0.2753370872811979) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.2753370872811979) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.2753370872811979) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.2753370872811979) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.2753370872811979) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.2753370872811979) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.2753370872811979) q[8];
cx q[6],q[8];
h q[0];
rz(0.9455932111289056) q[0];
h q[0];
h q[1];
rz(0.9455932111289056) q[1];
h q[1];
h q[2];
rz(0.9455932111289056) q[2];
h q[2];
h q[3];
rz(0.9455932111289056) q[3];
h q[3];
h q[4];
rz(0.9455932111289056) q[4];
h q[4];
h q[5];
rz(0.9455932111289056) q[5];
h q[5];
h q[6];
rz(0.9455932111289056) q[6];
h q[6];
h q[7];
rz(0.9455932111289056) q[7];
h q[7];
h q[8];
rz(0.9455932111289056) q[8];
h q[8];
h q[9];
rz(0.9455932111289056) q[9];
h q[9];
cx q[0],q[4];
u1(0.18717925966432514) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.18717925966432514) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.18717925966432514) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.18717925966432514) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.18717925966432514) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.18717925966432514) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.18717925966432514) q[8];
cx q[6],q[8];
h q[0];
rz(0.013049413774888441) q[0];
h q[0];
h q[1];
rz(0.013049413774888441) q[1];
h q[1];
h q[2];
rz(0.013049413774888441) q[2];
h q[2];
h q[3];
rz(0.013049413774888441) q[3];
h q[3];
h q[4];
rz(0.013049413774888441) q[4];
h q[4];
h q[5];
rz(0.013049413774888441) q[5];
h q[5];
h q[6];
rz(0.013049413774888441) q[6];
h q[6];
h q[7];
rz(0.013049413774888441) q[7];
h q[7];
h q[8];
rz(0.013049413774888441) q[8];
h q[8];
h q[9];
rz(0.013049413774888441) q[9];
h q[9];
cx q[0],q[4];
u1(0.33213763261491114) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.33213763261491114) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.33213763261491114) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.33213763261491114) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.33213763261491114) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.33213763261491114) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.33213763261491114) q[8];
cx q[6],q[8];
h q[0];
rz(0.4325690656230261) q[0];
h q[0];
h q[1];
rz(0.4325690656230261) q[1];
h q[1];
h q[2];
rz(0.4325690656230261) q[2];
h q[2];
h q[3];
rz(0.4325690656230261) q[3];
h q[3];
h q[4];
rz(0.4325690656230261) q[4];
h q[4];
h q[5];
rz(0.4325690656230261) q[5];
h q[5];
h q[6];
rz(0.4325690656230261) q[6];
h q[6];
h q[7];
rz(0.4325690656230261) q[7];
h q[7];
h q[8];
rz(0.4325690656230261) q[8];
h q[8];
h q[9];
rz(0.4325690656230261) q[9];
h q[9];
cx q[0],q[4];
u1(0.9162656681007448) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.9162656681007448) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.9162656681007448) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.9162656681007448) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.9162656681007448) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.9162656681007448) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.9162656681007448) q[8];
cx q[6],q[8];
h q[0];
rz(0.9592921245498005) q[0];
h q[0];
h q[1];
rz(0.9592921245498005) q[1];
h q[1];
h q[2];
rz(0.9592921245498005) q[2];
h q[2];
h q[3];
rz(0.9592921245498005) q[3];
h q[3];
h q[4];
rz(0.9592921245498005) q[4];
h q[4];
h q[5];
rz(0.9592921245498005) q[5];
h q[5];
h q[6];
rz(0.9592921245498005) q[6];
h q[6];
h q[7];
rz(0.9592921245498005) q[7];
h q[7];
h q[8];
rz(0.9592921245498005) q[8];
h q[8];
h q[9];
rz(0.9592921245498005) q[9];
h q[9];
cx q[0],q[4];
u1(0.27595712813025985) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.27595712813025985) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.27595712813025985) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.27595712813025985) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.27595712813025985) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.27595712813025985) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.27595712813025985) q[8];
cx q[6],q[8];
h q[0];
rz(1.8412342461847497) q[0];
h q[0];
h q[1];
rz(1.8412342461847497) q[1];
h q[1];
h q[2];
rz(1.8412342461847497) q[2];
h q[2];
h q[3];
rz(1.8412342461847497) q[3];
h q[3];
h q[4];
rz(1.8412342461847497) q[4];
h q[4];
h q[5];
rz(1.8412342461847497) q[5];
h q[5];
h q[6];
rz(1.8412342461847497) q[6];
h q[6];
h q[7];
rz(1.8412342461847497) q[7];
h q[7];
h q[8];
rz(1.8412342461847497) q[8];
h q[8];
h q[9];
rz(1.8412342461847497) q[9];
h q[9];
cx q[0],q[4];
u1(0.851337708730417) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.851337708730417) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.851337708730417) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.851337708730417) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.851337708730417) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.851337708730417) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.851337708730417) q[8];
cx q[6],q[8];
h q[0];
rz(1.841761652118173) q[0];
h q[0];
h q[1];
rz(1.841761652118173) q[1];
h q[1];
h q[2];
rz(1.841761652118173) q[2];
h q[2];
h q[3];
rz(1.841761652118173) q[3];
h q[3];
h q[4];
rz(1.841761652118173) q[4];
h q[4];
h q[5];
rz(1.841761652118173) q[5];
h q[5];
h q[6];
rz(1.841761652118173) q[6];
h q[6];
h q[7];
rz(1.841761652118173) q[7];
h q[7];
h q[8];
rz(1.841761652118173) q[8];
h q[8];
h q[9];
rz(1.841761652118173) q[9];
h q[9];
cx q[0],q[4];
u1(0.12514088419041325) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.12514088419041325) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.12514088419041325) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.12514088419041325) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.12514088419041325) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.12514088419041325) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.12514088419041325) q[8];
cx q[6],q[8];
h q[0];
rz(1.5470147882962832) q[0];
h q[0];
h q[1];
rz(1.5470147882962832) q[1];
h q[1];
h q[2];
rz(1.5470147882962832) q[2];
h q[2];
h q[3];
rz(1.5470147882962832) q[3];
h q[3];
h q[4];
rz(1.5470147882962832) q[4];
h q[4];
h q[5];
rz(1.5470147882962832) q[5];
h q[5];
h q[6];
rz(1.5470147882962832) q[6];
h q[6];
h q[7];
rz(1.5470147882962832) q[7];
h q[7];
h q[8];
rz(1.5470147882962832) q[8];
h q[8];
h q[9];
rz(1.5470147882962832) q[9];
h q[9];
cx q[0],q[4];
u1(0.7005547506835013) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.7005547506835013) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.7005547506835013) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.7005547506835013) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.7005547506835013) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.7005547506835013) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.7005547506835013) q[8];
cx q[6],q[8];
h q[0];
rz(1.5505260116695316) q[0];
h q[0];
h q[1];
rz(1.5505260116695316) q[1];
h q[1];
h q[2];
rz(1.5505260116695316) q[2];
h q[2];
h q[3];
rz(1.5505260116695316) q[3];
h q[3];
h q[4];
rz(1.5505260116695316) q[4];
h q[4];
h q[5];
rz(1.5505260116695316) q[5];
h q[5];
h q[6];
rz(1.5505260116695316) q[6];
h q[6];
h q[7];
rz(1.5505260116695316) q[7];
h q[7];
h q[8];
rz(1.5505260116695316) q[8];
h q[8];
h q[9];
rz(1.5505260116695316) q[9];
h q[9];
cx q[0],q[4];
u1(0.5675248578106579) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.5675248578106579) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.5675248578106579) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.5675248578106579) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.5675248578106579) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.5675248578106579) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.5675248578106579) q[8];
cx q[6],q[8];
h q[0];
rz(0.6897798078511552) q[0];
h q[0];
h q[1];
rz(0.6897798078511552) q[1];
h q[1];
h q[2];
rz(0.6897798078511552) q[2];
h q[2];
h q[3];
rz(0.6897798078511552) q[3];
h q[3];
h q[4];
rz(0.6897798078511552) q[4];
h q[4];
h q[5];
rz(0.6897798078511552) q[5];
h q[5];
h q[6];
rz(0.6897798078511552) q[6];
h q[6];
h q[7];
rz(0.6897798078511552) q[7];
h q[7];
h q[8];
rz(0.6897798078511552) q[8];
h q[8];
h q[9];
rz(0.6897798078511552) q[9];
h q[9];
cx q[0],q[4];
u1(0.998195098214822) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.998195098214822) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.998195098214822) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.998195098214822) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.998195098214822) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.998195098214822) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.998195098214822) q[8];
cx q[6],q[8];
h q[0];
rz(1.2451314680832013) q[0];
h q[0];
h q[1];
rz(1.2451314680832013) q[1];
h q[1];
h q[2];
rz(1.2451314680832013) q[2];
h q[2];
h q[3];
rz(1.2451314680832013) q[3];
h q[3];
h q[4];
rz(1.2451314680832013) q[4];
h q[4];
h q[5];
rz(1.2451314680832013) q[5];
h q[5];
h q[6];
rz(1.2451314680832013) q[6];
h q[6];
h q[7];
rz(1.2451314680832013) q[7];
h q[7];
h q[8];
rz(1.2451314680832013) q[8];
h q[8];
h q[9];
rz(1.2451314680832013) q[9];
h q[9];
cx q[0],q[4];
u1(0.8219535184225479) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.8219535184225479) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.8219535184225479) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.8219535184225479) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.8219535184225479) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.8219535184225479) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.8219535184225479) q[8];
cx q[6],q[8];
h q[0];
rz(1.294783761591035) q[0];
h q[0];
h q[1];
rz(1.294783761591035) q[1];
h q[1];
h q[2];
rz(1.294783761591035) q[2];
h q[2];
h q[3];
rz(1.294783761591035) q[3];
h q[3];
h q[4];
rz(1.294783761591035) q[4];
h q[4];
h q[5];
rz(1.294783761591035) q[5];
h q[5];
h q[6];
rz(1.294783761591035) q[6];
h q[6];
h q[7];
rz(1.294783761591035) q[7];
h q[7];
h q[8];
rz(1.294783761591035) q[8];
h q[8];
h q[9];
rz(1.294783761591035) q[9];
h q[9];
cx q[0],q[4];
u1(0.6177686086450798) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.6177686086450798) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.6177686086450798) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.6177686086450798) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.6177686086450798) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.6177686086450798) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.6177686086450798) q[8];
cx q[6],q[8];
h q[0];
rz(0.19350192631995444) q[0];
h q[0];
h q[1];
rz(0.19350192631995444) q[1];
h q[1];
h q[2];
rz(0.19350192631995444) q[2];
h q[2];
h q[3];
rz(0.19350192631995444) q[3];
h q[3];
h q[4];
rz(0.19350192631995444) q[4];
h q[4];
h q[5];
rz(0.19350192631995444) q[5];
h q[5];
h q[6];
rz(0.19350192631995444) q[6];
h q[6];
h q[7];
rz(0.19350192631995444) q[7];
h q[7];
h q[8];
rz(0.19350192631995444) q[8];
h q[8];
h q[9];
rz(0.19350192631995444) q[9];
h q[9];
cx q[0],q[4];
u1(0.13975317367490625) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.13975317367490625) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.13975317367490625) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.13975317367490625) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.13975317367490625) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.13975317367490625) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.13975317367490625) q[8];
cx q[6],q[8];
h q[0];
rz(0.9433837244583574) q[0];
h q[0];
h q[1];
rz(0.9433837244583574) q[1];
h q[1];
h q[2];
rz(0.9433837244583574) q[2];
h q[2];
h q[3];
rz(0.9433837244583574) q[3];
h q[3];
h q[4];
rz(0.9433837244583574) q[4];
h q[4];
h q[5];
rz(0.9433837244583574) q[5];
h q[5];
h q[6];
rz(0.9433837244583574) q[6];
h q[6];
h q[7];
rz(0.9433837244583574) q[7];
h q[7];
h q[8];
rz(0.9433837244583574) q[8];
h q[8];
h q[9];
rz(0.9433837244583574) q[9];
h q[9];
cx q[0],q[4];
u1(0.7649213235581586) q[4];
cx q[0],q[4];
cx q[0],q[5];
u1(0.7649213235581586) q[5];
cx q[0],q[5];
cx q[0],q[8];
u1(0.7649213235581586) q[8];
cx q[0],q[8];
cx q[1],q[6];
u1(0.7649213235581586) q[6];
cx q[1],q[6];
cx q[4],q[6];
u1(0.7649213235581586) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.7649213235581586) q[7];
cx q[4],q[7];
cx q[6],q[8];
u1(0.7649213235581586) q[8];
cx q[6],q[8];
h q[0];
rz(0.9464427319241882) q[0];
h q[0];
h q[1];
rz(0.9464427319241882) q[1];
h q[1];
h q[2];
rz(0.9464427319241882) q[2];
h q[2];
h q[3];
rz(0.9464427319241882) q[3];
h q[3];
h q[4];
rz(0.9464427319241882) q[4];
h q[4];
h q[5];
rz(0.9464427319241882) q[5];
h q[5];
h q[6];
rz(0.9464427319241882) q[6];
h q[6];
h q[7];
rz(0.9464427319241882) q[7];
h q[7];
h q[8];
rz(0.9464427319241882) q[8];
h q[8];
h q[9];
rz(0.9464427319241882) q[9];
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