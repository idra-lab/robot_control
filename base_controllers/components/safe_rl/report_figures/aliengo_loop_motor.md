168 episodes, the same in every arm; reference alone_motor.

| arm | failed | fall rule | paper fall | knees | joint limits | speed only | switched | walking | needless | stopped | prevented | caused | switch s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alone_motor | 50.0 | 8.3 | 10.1 | 35.7 | 4.8 | 2.4 | 0.0 | 0.0 | 0.0 | - | 0 | 0 | - |
| safe_motor | 18.5 | 3.6 | 1.2 | 12.5 | 1.2 | 1.2 | 100.0 | 0.0 | 100.0 | 66.1 | 54 | 1 | - |
| abs_motor | 14.3 | 6.0 | 0.6 | 7.7 | 0.6 | 0.6 | 67.3 | 0.0 | 38.1 | 61.1 | 61 | 1 | 0.20 |
| mps_motor | 14.3 | 4.2 | 0.0 | 7.7 | 3.0 | 2.4 | 78.0 | 0.0 | 57.1 | 58.0 | 60 | 0 | 0.24 |

Episode by episode against abs_motor (failures):

| arm | failed only with the arm | failed only with abs_motor | p |
|---|---|---|---|
| alone_motor | 61 | 1 | 2.73e-17 |
| safe_motor | 21 | 14 | 0.311 |
| mps_motor | 11 | 11 | 1 |
