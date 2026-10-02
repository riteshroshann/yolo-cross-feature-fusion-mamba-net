## Main results

| run | AP | AP50 | AP_small | AP_vt | AP_t | AP_s | params_M | GFLOPs | median_ms |
|---|---|---|---|---|---|---|---|---|---|
| llvip_yolo26n_visible | 0.5195 | 0.9000 | 0.0933 |  |  | 0.0933 | 2.5042 | 5.8922 | 17.2100 |
| llvip_yolo26n_thermal | 0.6409 | 0.9615 | 0.0642 |  |  | 0.0643 | 2.5042 | 5.8922 | 16.9800 |
| llvip_concat | 0.6495 | 0.9677 | 0.1274 |  |  | 0.1276 | 4.0673 | 9.7081 | 23.4440 |
| llvip_cffm | 0.6288 | 0.9575 | 0.0867 |  |  | 0.0867 | 6.0313 | 18.3875 | 41.9056 |
| llvip_cffm_nogate | 0.6258 | 0.9596 | 0.0645 |  |  | 0.0646 | 6.0088 | 18.1386 | 41.0870 |
| llvip_cffm_gconv | 0.6280 | 0.9599 | 0.0613 |  |  | 0.0616 | 6.1537 | 18.2952 | 34.6948 |
| m3fd_concat | 0.5048 | 0.7828 | 0.3149 | 0.0641 | 0.2215 | 0.4454 | 4.0692 | 9.7189 | 24.0353 |
| m3fd_cffm | 0.4791 | 0.7726 | 0.2978 | 0.0594 | 0.1908 | 0.4278 | 6.0326 | 18.4092 | 42.4553 |

## Hypotheses

| id | hypothesis | evidence | verdict |
|---|---|---|---|
| PH1 | fusion beats the best single sensor at night | -0.0120 AP vs thermal-only | inconclusive |
| PH2 | gating the step adds value | clean +0.0031 AP; degrades less by +0.0133 AP | inconclusive |
| PH3 | the pathway helps 8-32 px objects (M3FD) | -0.0242 AP in the 8-32 px bands | against |
| PH4 | the selective scan is needed | +0.0009 AP vs gated-conv control | inconclusive |
| PH5 | latency at most 2x the two-stream baseline | ratio 1.79 | supported |

## Degradation probe

| probe | llvip_cffm | llvip_cffm_nogate | llvip_concat |
|---|---|---|---|
| clean | 0.6288 | 0.6258 | 0.6495 |
| visible_dark | 0.5996 | 0.5421 | 0.6105 |
| visible_drop | 0.5673 | 0.5702 | 0.6184 |
| visible_drop (flagged) | 0.5950 | 0.5702 | 0.6184 |
| thermal_drop | 0.0217 | 0.0067 | 0.0029 |
| thermal_drop (flagged) | 0.1737 | 0.0067 | 0.0029 |
| thermal_shift_4 | 0.5327 | 0.5180 | 0.5157 |
| thermal_shift_8 | 0.3259 | 0.3110 | 0.2969 |
| thermal_shift_16 | 0.0276 | 0.0286 | 0.0169 |
