# Go2: value functions on the same trajectories

Walking policy alone, 238 pushes (116 ended in a failure): where each model's rule would switch (s after the push start); lead: the failure time minus the switch.

| model | level | of the failures, would switch | before the failure | median switch | median lead | of the pushes survived, would switch | before the push |
|---|---|---|---|---|---|---|---|
| abs | 0.050 (k 1) | 100.0 % | 100.0 % | 0.202 | 0.345 | 13.9 % | 0 |
| mps | 0.138 (k 3) | 96.6 % | 96.6 % | 0.294 | 0.230 | 7.4 % | 0 |
| mps_start | 0.503 (k 3) | 99.1 % | 99.1 % | 0.230 | 0.250 | 20.5 % | 0 |
| mps_walk | 0.999 (k 3) | 100.0 % | 100.0 % | 0.221 | 0.325 | 41.8 % | 0 |
| mps_start_walk | 0.878 (k 3) | 100.0 % | 100.0 % | 0.196 | 0.350 | 54.1 % | 0 |

On the 112 failures both abs and mps would switch in: mps later in 103, earlier in 5, median mps - abs 0.070 s (10-90 %: 0.021 to 0.230).
