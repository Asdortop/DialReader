| gauge type | parses tried | accepted by validation | accepted and correct | VLM answer correct without validation | label-to-tick match rate % | rejection reasons |
| --- | --- | --- | --- | --- | --- | --- |
| pressure_gauge | 3 | 1 | 1 | 3 | 57.60 | too few labels line up with detected major ticks |
| tyre_gauge | 3 | 2 | 2 | 3 | 81.00 | too few labels line up with detected major ticks |
| kitchen_scale | 3 | 2 | 2 | 3 | 77.80 | too few labels line up with detected major ticks |
| thermometer | 3 | 2 | 2 | 3 | 83.30 | too few labels line up with detected major ticks |
| hygrometer | 3 | 2 | 2 | 3 | 93.90 | labelled values are not evenly spaced along the ticks |
| barometer | 3 | 1 | 1 | 3 | 60.60 | too few labels line up with detected major ticks |
| oven_thermometer | 3 | 1 | 1 | 1 | 61.10 | too few labels line up with detected major ticks |
| vacuum_gauge | 3 | 2 | 2 | 2 | 66.70 | scale type 'nonlinear' is out of scope |
| voltmeter | 3 | 2 | 2 | 3 | 81.00 | labelled values are not evenly spaced along the ticks |
| speedometer | 3 | 3 | 3 | 3 | 100.00 |  |
| megohmmeter | 3 | 1 | 1 | 3 | 66.70 | label values are not monotonic along the scale; too few labels line up with detected major ticks |
