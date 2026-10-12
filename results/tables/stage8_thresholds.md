| signal | meaning when crossed | worse when | default | tuned on validation |
| --- | --- | --- | --- | --- |
| needle_spread | needle angle is uncertain | larger | 1.00 | 1.08 |
| method_gap | line fit and Hough line disagree on the needle | larger | 1.50 | 0.55 |
| rim_support | dial rim only partly found | smaller | 0.55 | 0.41 |
| rim_residual | dial rim does not fit an ellipse well | larger | 1.20 | 1.18 |
| comb_residual | tick marks are not evenly spaced | larger | 0.12 | 0.08 |
| comb_missing | many tick marks are missing | larger | 0.15 | 0.07 |
| dft_gap | tick spacing from the DFT disagrees with the comb | larger | 0.06 | 0.01 |
| tick_snr | tick marks are faint | smaller | 6.00 | 11.65 |
| glare_overlap | glare covers the needle | larger | 0.25 | 0.59 |
| tick_count | tick count does not match the scale | larger | 0.50 | 0.50 |
| out_of_range | needle is outside the scale | larger | 1.00 | 1.00 |
| sentinel | VLM coarse reading disagrees (gross-error sentinel) | larger | 1.00 | 1.00 |
