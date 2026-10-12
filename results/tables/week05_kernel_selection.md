| kernel | variance of Laplacian | gradient L1/L2 | PSNR vs sharp photo (dB) | chosen by |
| --- | --- | --- | --- | --- |
| none (input) | 70.58 | 0.29 | 29.50 |  |
| gaussian sigma=1.0 | 233.14 | 0.46 | 30.40 |  |
| gaussian sigma=1.5 | 259.75 | 0.39 | 32.48 |  |
| gaussian sigma=2.0 | 551.56 | 0.33 | 35.69 |  best PSNR (truth) |
| gaussian sigma=3.0 | 1604.37 | 0.31 | 28.19 | highest Laplacian variance |
| motion len=7 angle=0 | 462.10 | 0.52 | 29.83 |  |
| motion len=7 angle=45 | 512.27 | 0.49 | 29.82 |  |
| motion len=7 angle=90 | 505.91 | 0.51 | 29.94 |  |
| motion len=7 angle=135 | 468.04 | 0.50 | 29.89 |  |
| motion len=11 angle=0 | 972.00 | 0.52 | 24.10 |  |
| motion len=11 angle=45 | 1140.97 | 0.51 | 22.04 |  |
| motion len=11 angle=90 | 1115.08 | 0.51 | 23.24 |  |
| motion len=11 angle=135 | 1039.55 | 0.53 | 22.68 |  |
| motion len=15 angle=0 | 1167.73 | 0.50 | 21.36 |  |
| motion len=15 angle=45 | 1276.87 | 0.50 | 20.11 |  |
| motion len=15 angle=90 | 1232.04 | 0.49 | 20.86 |  |
| motion len=15 angle=135 | 1153.05 | 0.51 | 20.86 |  |
| edge-based estimate: gaussian sigma=1.98 |  |  | 35.75 | used by Stage 1 |
