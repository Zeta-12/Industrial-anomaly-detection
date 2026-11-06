| Variant | Threshold | Image AUROC | AUPRO | Recall @cal. | FPR @cal. | Spearman vs PyTorch | Stage 1 p50 (ms) | Stage 2 p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| FP32 features + FP32 bank | 35.16 | 0.949 | 0.940 | 0.908 | 0.115 | 1.000 | 84.0 | 325.7 |
| INT8 features + FP32 bank | 38.79 | 0.935 | 0.882 | 0.887 | 0.077 | 0.971 | 22.3 | 324.5 |
| INT8 features + INT8 bank | 35.83 | 0.944 | 0.934 | 0.901 | 0.115 | 0.980 | 22.6 | 331.0 |
