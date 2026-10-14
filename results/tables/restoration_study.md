| degradation | photos | PSNR degraded (dB) | PSNR after Stage 1 (dB) | read within 1 division %, Stage 1 on | read within 1 division %, Stage 1 off | steps applied |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 36 |  | 36.50 | 100.00 | 100.00 | sharpen x36, contrast x8 |
| Gaussian blur 2 px | 36 | 31.30 | 34.20 | 100.00 | 100.00 | deblur x36, contrast x9 |
| Gaussian blur 4 px | 36 | 27.00 | 28.30 | 100.00 | 77.80 | deblur x36, contrast x10 |
| motion blur 13 px | 36 | 28.40 | 27.90 | 94.40 | 63.90 | deblur x35, contrast x7, sharpen x1 |
| Gaussian noise 20 | 36 | 22.30 | 28.10 | 100.00 | 100.00 | denoise x36, contrast x10 |
| salt-and-pepper 6% | 36 | 17.30 | 34.10 | 100.00 | 83.30 | denoise x36, contrast x9 |
| blur 2 px + noise 12 | 36 | 25.30 | 27.40 | 100.00 | 97.20 | denoise x36, contrast x10 |
