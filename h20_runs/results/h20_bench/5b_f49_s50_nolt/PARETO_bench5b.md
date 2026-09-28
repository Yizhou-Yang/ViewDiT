
### speed 1.3-1.7x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| FasterCache | FC_v2 | 1.43 | 28.48 | 0.9586 | 0.0390 | -0.04 |
| PAB | PAB_k2 | 1.36 | 28.36 | 0.9580 | 0.0396 | +0.01 |
| TeaCache | TEA_0.1 | 1.51 | 21.24 | 0.8649 | 0.1283 | +0.03 |

### speed 1.7-2.05x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_Ow20_L4 | 1.78 | 31.65 | 0.9730 | 0.0221 | +0.12 |
| TeaCache | TEA_0.15 | 2.00 | 19.32 | 0.7970 | 0.1801 | -0.24 |
| FORA | FORA_n2 | 1.85 | 16.85 | 0.7117 | 0.2616 | -0.11 |
| steps | steps25 | 2.00 | 11.33 | 0.5614 | 0.5443 | -0.74 |

### speed 2.05-2.45x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_Ow17_L4_GI | 2.07 | 28.53 | 0.9525 | 0.0379 | +0.12 |
| TeaCache | TEA_0.2 | 2.38 | 18.62 | 0.7647 | 0.2150 | -0.00 |

### speed 2.45-2.85x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O3w13_B2 | 2.50 | 23.61 | 0.8993 | 0.0851 | +0.05 |
| TaylorSeer | TS_n3 | 2.47 | 14.94 | 0.6497 | 0.3426 | -0.05 |
| steps | steps20 | 2.50 | 10.94 | 0.5544 | 0.5676 | -0.67 |

### speed 2.85-3.2x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O4w13_B2_GI | 2.93 | 22.68 | 0.8790 | 0.1007 | +0.08 |
| TeaCache | TEA_0.3 | 3.11 | 16.79 | 0.6961 | 0.2829 | -0.20 |

### speed 3.2-4.0x

| family | best method | speed | PSNR | SSIM | LPIPS | dCLIP |
|---|---|---|---|---|---|---|
| OURS | OURS_O3w10_B3 | 3.33 | 20.46 | 0.8232 | 0.1592 | +0.22 |

### Pareto front per family (speed, PSNR)

- FORA: FORA_n2(1.85x,16.9)
- FasterCache: FC_v2(1.43x,28.5)
- OURS: OURS_O4w10_B2(3.34x,20.2), OURS_O3w10_B3(3.33x,20.5), OURS_O4w13_B2_GI(2.93x,22.7), OURS_O3w13_B2(2.50x,23.6), OURS_O3w17_B2(2.17x,27.1), OURS_Ow17_L4_GI(2.07x,28.5), OURS_Ow20_L4(1.78x,31.7)
- PAB: PAB_k3(1.52x,25.4), PAB_k2(1.36x,28.4)
- TaylorSeer: TS_n3(2.47x,14.9)
- TeaCache: TEA_0.3(3.11x,16.8), TEA_0.2(2.38x,18.6), TEA_0.15(2.00x,19.3), TEA_0.1(1.51x,21.2)
- ref: full_perturb1e-2(1.00x,23.9)
- steps: steps20(2.50x,10.9), steps25(2.00x,11.3)
