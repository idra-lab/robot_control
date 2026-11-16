# Aliengo: value functions on the same trajectories

Walking policy alone, 240 pushes (106 ended in a failure): where each model's rule would switch (s after the push start); lead: the failure time minus the switch.

| model | level | of the failures, would switch | before the failure | median switch | median lead | of the pushes survived, would switch | before the push |
|---|---|---|---|---|---|---|---|
| abs | 0.050 (k 1) | 100.0 % | 100.0 % | 0.186 | 0.310 | 36.6 % | 0 |
| mps | 0.919 (k 3) | 98.1 % | 98.1 % | 0.198 | 0.310 | 60.4 % | 1 |
| mps_start | -0.038 (k 3) | 9.4 % | 9.4 % | 0.308 | 0.255 | 0.0 % | 0 |

On the 104 failures both abs and mps would switch in: mps later in 63, earlier in 32, median mps - abs 0.020 s (10-90 %: -0.050 to 0.107).

abs acting, 214 pushes, switched in 132 (26 of them failed): each model's V at the acting filter's switch (median), and the share below its level.

| model | level | V at the switch, then failed | below the level | V at the switch, then recovered | below the level |
|---|---|---|---|---|---|
| abs | 0.050 | -0.043 | 100 % | -0.027 | 99 % |
| mps | 0.919 | 0.877 | 65 % | 0.921 | 48 % |
| mps_start | -0.038 | 0.753 | 0 % | 0.788 | 1 % |

mps acting, 233 pushes, switched in 180 (44 of them failed): each model's V at the acting filter's switch (median), and the share below its level.

| model | level | V at the switch, then failed | below the level | V at the switch, then recovered | below the level |
|---|---|---|---|---|---|
| abs | 0.050 | -0.276 | 68 % | 0.405 | 35 % |
| mps | 0.919 | 0.833 | 100 % | 0.857 | 99 % |
| mps_start | -0.038 | 0.688 | 5 % | 0.841 | 0 % |
