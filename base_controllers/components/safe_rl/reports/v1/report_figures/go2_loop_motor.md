156 episodes, the same in every arm; reference alone_motor.

| arm | failed | fall rule | paper fall | knees | joint limits | speed only | switched | walking | needless | stopped | prevented | caused | switch s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alone_motor | 46.8 | 32.7 | 9.6 | 7.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | 0 | 0 | - |
| safe_motor | 32.7 | 10.9 | 1.3 | 21.2 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 | 52.6 | 44 | 22 | - |
| abs_motor | 21.2 | 10.9 | 0.6 | 10.3 | 0.0 | 0.0 | 55.8 | 0.0 | 19.3 | 39.1 | 41 | 1 | 0.21 |
| mps_motor | 27.6 | 17.9 | 3.2 | 8.3 | 0.0 | 0.0 | 67.9 | 0.6 | 39.8 | 31.1 | 36 | 6 | 0.24 |

Episode by episode against abs_motor (failures):

| arm | failed only with the arm | failed only with abs_motor | p |
|---|---|---|---|
| alone_motor | 41 | 1 | 1.96e-11 |
| safe_motor | 35 | 17 | 0.0175 |
| mps_motor | 16 | 6 | 0.0525 |
