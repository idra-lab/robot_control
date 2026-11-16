# Aliengo: value functions on the same trajectories

Walking policy alone, 238 pushes (107 ended in a failure): where each model's rule would switch (s after the push start); lead: the failure time minus the switch.

| model | level | of the failures, would switch | before the failure | median switch | median lead | of the pushes survived, would switch | before the push |
|---|---|---|---|---|---|---|---|
| abs | 0.050 (k 1) | 100.0 % | 100.0 % | 0.184 | 0.330 | 29.8 % | 0 |
| mps | 0.919 (k 3) | 96.3 % | 96.3 % | 0.204 | 0.340 | 62.6 % | 2 |
| mps_start | -0.038 (k 3) | 14.0 % | 14.0 % | 0.352 | 0.190 | 0.8 % | 1 |
| mps_walk | 0.981 (k 3) | 100.0 % | 100.0 % | 0.168 | 0.350 | 88.5 % | 2 |
| mps_start_walk | 0.917 (k 3) | 94.4 % | 94.4 % | 0.194 | 0.320 | 45.8 % | 2 |

On the 103 failures both abs and mps would switch in: mps later in 64, earlier in 32, median mps - abs 0.020 s (10-90 %: -0.040 to 0.100).
