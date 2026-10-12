| data | id | what is switched on | n | answered % | coverage % (reliable) | mean error (% FS) | 90th percentile | gross-error rate % | gross-error rate %, all answered | coverage % at risk <= 2% | useful readings % (reliable and correct) | silent failures % (reliable but gross) | VLM calls per image | vs previous step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Test split | C0 | Classical pipeline, scale entered by hand | 182 | 97.80 | 89.56 | 0.02 | 0.04 | 0.00 | 0.56 | 97.80 | 89.56 | 0.00 | 0.00 |  |
| Test split | C1 | C0 + hook 1 (scale parser) | 182 | 97.80 | 89.56 | 0.02 | 0.04 | 0.00 | 0.56 | 97.80 | 89.56 | 0.00 | 0.11 | no measurable effect: useful +0.0 [+0.0, +0.0], silent failures +0.0 [+0.0, +0.0] |
| Test split | C2 | C1 + hook 2 (sentinel flags) | 182 | 97.80 | 86.26 | 0.02 | 0.04 | 0.00 | 0.56 | 97.80 | 86.26 | 0.00 | 1.11 | hurts: useful -3.3 [-6.0, -1.1], silent failures +0.0 [+0.0, +0.0] |
| Test split | C3 | C2 + hook 3 (repair loop) | 182 | 98.90 | 87.91 | 0.02 | 0.04 | 0.00 | 1.67 | 98.90 | 87.91 | 0.00 | 1.37 | no measurable effect: useful +1.6 [+0.0, +3.8], silent failures +0.0 [+0.0, +0.0] |
| Test split | B | VLM alone, as the baseline | 182 | 100.00 | 100.00 | 1.84 | 2.81 | 1.65 | 1.65 | 100.00 | 85.71 | 1.65 | 1.00 |  |
| Stress set | C0 | Classical pipeline, scale entered by hand | 182 | 90.11 | 75.82 | 0.03 | 0.07 | 0.00 | 6.10 | 83.52 | 75.82 | 0.00 | 0.00 |  |
| Stress set | C1 | C0 + hook 1 (scale parser) | 182 | 90.11 | 75.82 | 0.03 | 0.07 | 0.00 | 6.10 | 83.52 | 75.82 | 0.00 | 0.11 | no measurable effect: useful +0.0 [+0.0, +0.0], silent failures +0.0 [+0.0, +0.0] |
| Stress set | C2 | C1 + hook 2 (sentinel flags) | 182 | 90.11 | 53.85 | 0.03 | 0.06 | 0.00 | 6.10 | 71.43 | 53.85 | 0.00 | 1.11 | hurts: useful -22.0 [-28.0, -15.4], silent failures +0.0 [+0.0, +0.0] |
| Stress set | C3 | C2 + hook 3 (repair loop) | 182 | 91.21 | 56.59 | 0.76 | 0.07 | 0.97 | 11.45 | 58.79 | 56.04 | 0.55 | 2.03 | no measurable effect: useful +2.2 [+0.5, +4.4], silent failures +0.5 [+0.0, +1.6] |
| Stress set | B | VLM alone, as the baseline | 182 | 100.00 | 100.00 | 14.61 | 58.49 | 26.92 | 26.92 | 45.60 | 61.54 | 26.92 | 1.00 |  |
