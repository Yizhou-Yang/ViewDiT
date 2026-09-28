# Fairness round `fair` in ../results/h20_bench/2b_f49_s50_fair

quality key = psnr_lmse (psnr_lmse = -10log10 latent MSE to full, used when pixel scores missing)

## All methods

| method | fam | n | speed | PSNR | LPIPS | latPSNR | dAesth | dCLIP | dSubj | imaging× | motion× | flicker× |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEAw10_0.8 | TEAw | 4 | 3.81 | nan | nan | 8.49 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.5_GI | TEAw_GI | 4 | 3.54 | nan | nan | 10.01 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.5 | TEAw | 4 | 3.31 | nan | nan | 10.08 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.3 | TEAw | 4 | 2.93 | nan | nan | 12.46 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.2_GI | TEAw_GI | 4 | 2.75 | nan | nan | 14.19 | nan | nan | nan | nan | nan | nan |
| TEAw17_0.5_GI | TEAw_GI | 1 | 2.55 | nan | nan | 13.74 | nan | nan | nan | nan | nan | nan |
| dpm20 | dpm | 1 | 2.51 | nan | nan | 0.94 | nan | nan | nan | nan | nan | nan |
| unipc20 | unipc | 1 | 2.51 | nan | nan | 1.70 | nan | nan | nan | nan | nan | nan |
| FORAw10_n4 | FORAw | 1 | 2.49 | nan | nan | 18.13 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.2 | TEAw | 4 | 2.49 | nan | nan | 14.40 | nan | nan | nan | nan | nan | nan |
| FORAw10_n3 | FORAw | 4 | 2.08 | nan | nan | 14.64 | nan | nan | nan | nan | nan | nan |
| TEAw10_0.1 | TEAw | 4 | 1.85 | nan | nan | 17.95 | nan | nan | nan | nan | nan | nan |
| dpm30 | dpm3 | 1 | 1.67 | nan | nan | 2.75 | nan | nan | nan | nan | nan | nan |
| FORAw10_n2 | FORAw | 4 | 1.67 | nan | nan | 18.15 | nan | nan | nan | nan | nan | nan |
| TSw10_n2 | TSw | 1 | 1.66 | nan | nan | 25.82 | nan | nan | nan | nan | nan | nan |
| Oonly_O2w17 | Oonly | 1 | 1.48 | nan | nan | 35.61 | nan | nan | nan | nan | nan | nan |
| full_perturb1e-2 | full_perturb1e-2 | 4 | 1.00 | nan | nan | 17.67 | nan | nan | nan | nan | nan | nan |
| full | full | 4 | 1.00 | nan | nan | 120.00 | nan | nan | nan | nan | nan | nan |

## Best of each family per speed tier (by quality key)

### 1.3-1.65x
- Oonly_O2w17 (Oonly): 1.48x, psnr_lmse 35.61, LPIPS nan, dAesth nan, dCLIP nan

### 1.65-1.95x
- TSw10_n2 (TSw): 1.66x, psnr_lmse 25.82, LPIPS nan, dAesth nan, dCLIP nan
- FORAw10_n2 (FORAw): 1.67x, psnr_lmse 18.15, LPIPS nan, dAesth nan, dCLIP nan
- TEAw10_0.1 (TEAw): 1.85x, psnr_lmse 17.95, LPIPS nan, dAesth nan, dCLIP nan
- dpm30 (dpm3): 1.67x, psnr_lmse 2.75, LPIPS nan, dAesth nan, dCLIP nan

### 1.95-2.25x
- FORAw10_n3 (FORAw): 2.08x, psnr_lmse 14.64, LPIPS nan, dAesth nan, dCLIP nan

### 2.25-2.7x
- FORAw10_n4 (FORAw): 2.49x, psnr_lmse 18.13, LPIPS nan, dAesth nan, dCLIP nan
- TEAw10_0.2 (TEAw): 2.49x, psnr_lmse 14.40, LPIPS nan, dAesth nan, dCLIP nan
- TEAw17_0.5_GI (TEAw_GI): 2.55x, psnr_lmse 13.74, LPIPS nan, dAesth nan, dCLIP nan
- unipc20 (unipc): 2.51x, psnr_lmse 1.70, LPIPS nan, dAesth nan, dCLIP nan
- dpm20 (dpm): 2.51x, psnr_lmse 0.94, LPIPS nan, dAesth nan, dCLIP nan

### 2.7-3.1x
- TEAw10_0.2_GI (TEAw_GI): 2.75x, psnr_lmse 14.19, LPIPS nan, dAesth nan, dCLIP nan
- TEAw10_0.3 (TEAw): 2.93x, psnr_lmse 12.46, LPIPS nan, dAesth nan, dCLIP nan

### 3.1-3.6x
- TEAw10_0.5 (TEAw): 3.31x, psnr_lmse 10.08, LPIPS nan, dAesth nan, dCLIP nan
- TEAw10_0.5_GI (TEAw_GI): 3.54x, psnr_lmse 10.01, LPIPS nan, dAesth nan, dCLIP nan

### 3.6-5.0x
- TEAw10_0.8 (TEAw): 3.81x, psnr_lmse 8.49, LPIPS nan, dAesth nan, dCLIP nan

