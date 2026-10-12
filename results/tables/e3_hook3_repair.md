| data | method | flagged images | fixed (reliable and correct) | fixed % | made reliable but wrong | pipeline reruns per image | VLM calls per image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test split | No repair | 25 | 0 | 0.00 | 0 | 0.00 | 0.00 |
| Test split | Hook 3: VLM picks the action (at most 2 reruns) | 25 | 3 | 12.00 | 0 | 1.88 | 1.88 |
| Test split | Rule-based, fixed order (at most 2 reruns) | 25 | 3 | 12.00 | 0 | 1.96 | 0.00 |
| Test split | Grid search over all actions (5 reruns) | 25 | 3 | 12.00 | 1 | 5.00 | 0.00 |
| Stress set | No repair | 84 | 0 | 0.00 | 0 | 0.00 | 0.00 |
| Stress set | Hook 3: VLM picks the action (at most 2 reruns) | 84 | 4 | 4.76 | 1 | 1.98 | 1.98 |
| Stress set | Rule-based, fixed order (at most 2 reruns) | 84 | 3 | 3.57 | 0 | 2.00 | 0.00 |
| Stress set | Grid search over all actions (5 reruns) | 84 | 4 | 4.76 | 3 | 5.00 | 0.00 |
