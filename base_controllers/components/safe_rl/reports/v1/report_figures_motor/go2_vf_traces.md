# Go2: value functions on the same trajectories

Walking policy alone, 239 pushes (113 ended in a failure): where each model's rule would switch (s after the push start); lead: the failure time minus the switch.

| model | level | of the failures, would switch | before the failure | median switch | median lead | of the pushes survived, would switch | before the push |
|---|---|---|---|---|---|---|---|
| abs | 0.050 (k 1) | 100.0 % | 100.0 % | 0.204 | 0.340 | 17.5 % | 0 |
| mps | 0.138 (k 3) | 98.2 % | 98.2 % | 0.294 | 0.240 | 7.1 % | 0 |
| mps_start | 0.503 (k 3) | 99.1 % | 99.1 % | 0.242 | 0.260 | 18.3 % | 0 |

On the 111 failures both abs and mps would switch in: mps later in 103, earlier in 4, median mps - abs 0.070 s (10-90 %: 0.020 to 0.270).

abs acting, 194 pushes, switched in 96 (41 of them failed): each model's V at the acting filter's switch (median), and the share below its level.

| model | level | V at the switch, then failed | below the level | V at the switch, then recovered | below the level |
|---|---|---|---|---|---|
| abs | 0.050 | -0.012 | 100 % | -0.003 | 100 % |
| mps | 0.138 | 0.960 | 27 % | 0.999 | 11 % |
| mps_start | 0.503 | 0.459 | 51 % | 0.688 | 27 % |

mps acting, 231 pushes, switched in 118 (84 of them failed): each model's V at the acting filter's switch (median), and the share below its level.

| model | level | V at the switch, then failed | below the level | V at the switch, then recovered | below the level |
|---|---|---|---|---|---|
| abs | 0.050 | -0.506 | 95 % | -0.338 | 76 % |
| mps | 0.138 | 0.010 | 100 % | 0.030 | 100 % |
| mps_start | 0.503 | 0.273 | 87 % | 0.471 | 53 % |
