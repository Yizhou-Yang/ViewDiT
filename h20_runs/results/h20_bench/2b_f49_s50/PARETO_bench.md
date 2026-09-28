
### speed 1.3-1.7x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| PAB | PAB_k2 | 1.40 | 34.15 | 0.9665 | 0.0330 | -0.01 |
| FasterCache | FC_v2 | 1.45 | 34.12 | 0.9664 | 0.0332 | +0.00 |

### speed 1.7-2.05x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_Ow20_L4 | 1.79 | 37.82 | 0.9863 | 0.0143 | +0.00 |
| TeaCache | TEA_0.1 | 1.85 | 31.71 | 0.9531 | 0.0464 | -0.10 |
| FORA | FORA_n2 | 1.85 | 20.81 | 0.8101 | 0.1995 | -0.20 |
| steps | steps25 | 2.00 | 14.85 | 0.6398 | 0.4257 | -0.80 |

### speed 2.05-2.45x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_Ow17_L4_GI | 2.07 | 33.44 | 0.9692 | 0.0314 | -0.07 |
| TeaCache | TEA_0.15 | 2.37 | 23.40 | 0.8550 | 0.1496 | -0.15 |

### speed 2.45-2.85x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O3w13_B2 | 2.50 | 29.41 | 0.9368 | 0.0630 | -0.17 |
| TaylorSeer | TS_n3 | 2.46 | 20.31 | 0.7885 | 0.2237 | +0.01 |
| FORA | FORA_n3 | 2.62 | 19.13 | 0.7539 | 0.2632 | -0.66 |
| steps | steps20 | 2.50 | 13.68 | 0.5920 | 0.4854 | -1.35 |

### speed 2.85-3.2x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O4w13_B2_GI | 2.93 | 27.66 | 0.9185 | 0.0855 | -0.27 |
| TeaCache | TEA_0.2 | 2.92 | 21.92 | 0.8198 | 0.1839 | -0.26 |
| TaylorSeer | TS_n4 | 2.88 | 17.93 | 0.7178 | 0.2887 | +0.02 |
| steps | steps17 | 2.95 | 13.60 | 0.5921 | 0.5069 | -1.56 |

### speed 3.2-4.0x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O3w10_B3 | 3.33 | 25.00 | 0.8732 | 0.1311 | -0.08 |
| TeaCache | TEA_0.3 | 3.80 | 20.33 | 0.7846 | 0.2283 | -0.72 |

### Pareto front per family (speed, PSNR)

- FORA: FORA_n3(2.62x,19.1), FORA_n2(1.85x,20.8)
- FasterCache: FC_v2_k3(1.68x,30.9), FC_v2(1.45x,34.1)
- OURS: OURS_Ow7_L6_GI(3.67x,23.1), OURS_O4w10_B2(3.33x,24.8), OURS_O3w10_B3(3.33x,25.0), OURS_O5w13_B2(2.94x,26.6), OURS_O4w13_B2_GI(2.93x,27.7), OURS_O4w13_B2(2.78x,27.9), OURS_O3w13_B2(2.50x,29.4), OURS_O3w17_B2_GI(2.27x,31.9), OURS_O3w17_B2(2.18x,32.6), OURS_Ow17_L4_GI(2.07x,33.4), OURS_Ow17_L4(1.92x,35.2), OURS_Ow20_L4(1.79x,37.8)
- PAB: PAB_k3(1.58x,30.9), PAB_k2(1.40x,34.2)
- TaylorSeer: TS_n4(2.88x,17.9), TS_n3(2.46x,20.3)
- TeaCache: TEA_0.3(3.80x,20.3), TEA_0.2(2.92x,21.9), TEA_0.15(2.37x,23.4), TEA_0.1(1.85x,31.7)
- ref: full_perturb1e-2(1.00x,31.6)
- steps: steps17(2.95x,13.6), steps20(2.50x,13.7), steps25(2.00x,14.9)
