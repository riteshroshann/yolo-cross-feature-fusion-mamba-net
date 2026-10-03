## Main results

| run | AP | AP50 | AP_small | AP_vt | AP_t | AP_s | params_M | GFLOPs | median_ms |
|---|---|---|---|---|---|---|---|---|---|
| llvip_yolo26n_visible | 0.5195 | 0.8999 | 0.0933 |  |  | 0.0933 | 2.5042 | 5.8922 | 17.2100 |
| llvip_yolo26n_thermal | 0.6409 | 0.9615 | 0.0642 |  |  | 0.0643 | 2.5042 | 5.8922 | 16.9800 |
| llvip_concat | 0.6495 | 0.9676 | 0.1274 |  |  | 0.1276 | 4.0673 | 9.7081 | 23.4440 |
| llvip_cffm | 0.6289 | 0.9576 | 0.0885 |  |  | 0.0885 | 6.0313 | 18.3875 | 41.9056 |
| llvip_cffm_nogate | 0.6257 | 0.9596 | 0.0645 |  |  | 0.0645 | 6.0088 | 18.1386 | 41.0870 |
| llvip_cffm_gconv | 0.6279 | 0.9597 | 0.0612 |  |  | 0.0615 | 6.1537 | 18.2952 | 34.6948 |
| m3fd_concat | 0.5047 | 0.7820 | 0.3149 | 0.0641 | 0.2215 | 0.4455 | 4.0692 | 9.7189 | 24.0353 |
| m3fd_cffm | 0.4790 | 0.7726 | 0.2974 | 0.0595 | 0.1899 | 0.4275 | 6.0326 | 18.4092 | 42.4553 |
| llvip_cffm_nop2b | 0.6350 | 0.9604 | 0.0750 |  |  | 0.0750 |  |  |  |
| llvip_concat_p2 | 0.6321 | 0.9622 | 0.0556 |  |  | 0.0557 |  |  |  |

## Hypotheses

| id | hypothesis | evidence | verdict |
|---|---|---|---|
| PH1 | fusion beats the best single sensor at night | -0.0120 AP vs thermal-only | inconclusive |
| PH2 | gating the step adds value | clean +0.0032 AP; degrades less by +0.0130 AP | inconclusive |
| PH3 | the pathway helps 8-32 px objects (M3FD) | -0.0248 AP in the 8-32 px bands | against |
| PH4 | the selective scan is needed | +0.0010 AP vs gated-conv control | inconclusive |
| PH5 | latency at most 2x the two-stream baseline | ratio 1.79 | supported |

## Degradation probe

| probe | llvip_cffm | llvip_cffm_nogate | llvip_concat |
|---|---|---|---|
| clean | 0.6289 | 0.6258 | 0.6495 |
| visible_dark | 0.5985 | 0.5424 | 0.6099 |
| visible_drop | 0.5674 | 0.5704 | 0.6184 |
| visible_drop (flagged) | 0.5950 | 0.5704 | 0.6184 |
| thermal_drop | 0.0217 | 0.0066 | 0.0029 |
| thermal_drop (flagged) | 0.1739 | 0.0066 | 0.0029 |
| thermal_shift_4 | 0.5326 | 0.5180 | 0.5157 |
| thermal_shift_8 | 0.3259 | 0.3110 | 0.2969 |
| thermal_shift_16 | 0.0276 | 0.0286 | 0.0169 |
