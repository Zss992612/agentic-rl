# SFT dataset checkpoint-five overview

This report is a transient audit. No final token or label arrays were written.

## Dataset

- Trajectories: 256
- Failed trajectories: 0
- Unique tasks: 64
- Trajectories per task distribution: `{6: 8, 4: 9, 5: 9, 3: 15, 2: 16, 7: 6, 8: 1}`
- Total assistant spans: 3344
- Assistant text spans: 1280
- Assistant tool-call spans: 2064
- Tool results: 2064

## Token lengths

- Total tokens: 2,637,872
- Min / median / mean / max: 6,145 / 9,907.0 / 10,304.2 / 16,904
- P90 / P95 / P99: 13,527.0 / 13,971.2 / 15,710.8
- Within 16,384: 254
- Over 16,384: 2
- Length bins: `{'04097-08192': 29, '08193-12288': 187, '12289-16384': 38, '>16384': 2}`

## Supervision

- Total supervised tokens: 414,294
- Total masked tokens: 2,223,578
- Weighted supervised fraction: 15.71%
- Supervised min / median / mean / max: 624 / 1,425.0 / 1,618.3 / 5,995

## Assistant boundary alignment

- Trajectories with merged boundaries: 229
- Merged assistant spans: 355
- Masked leading characters: 710
- Non-whitespace characters masked at boundaries: 0

## Ten longest trajectories

| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |
|---:|---:|---:|---:|---:|---|
| 108 | 99 | 16904 | 5995 | 35.46% | 7cd38137-61b3-4f4e-8da3-3d54bba014cd |
| 107 | 98 | 16897 | 4058 | 24.02% | 056017b6-8da2-4a97-8500-89d59462db5d |
| 106 | 98 | 15839 | 4422 | 27.92% | 9bb35156-39ee-4b1b-8ee5-08bf494ebf80 |
| 201 | 99 | 15606 | 4602 | 29.49% | 110a383a-c797-4a33-8f82-68d80c9b31f8 |
| 244 | 99 | 15469 | 4530 | 29.28% | 5bc962ea-294f-4b16-a1e6-fee2e2c23570 |
| 135 | 98 | 15183 | 3583 | 23.60% | ea435c23-8668-43bc-8cf7-5af21e96fb6f |
| 31 | 20 | 15042 | 3286 | 21.85% | f6b05c85-5d14-4d39-a583-f4bd8da49190 |
| 228 | 99 | 14938 | 3965 | 26.54% | 0126f348-9917-4923-a1af-30f29d57081e |
| 233 | 98 | 14842 | 2097 | 14.13% | 2b1ba4de-cbd5-4bd6-b33d-055da4961b87 |
| 77 | 76 | 14831 | 3512 | 23.68% | 970ea165-9473-433f-807a-d057d40c46dc |

## Ten lowest supervision fractions

| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |
|---:|---:|---:|---:|---:|---|
| 126 | 113 | 8416 | 695 | 8.26% | 52d58da9-e37e-4882-b372-8b3a11e472c1 |
| 191 | 113 | 8427 | 710 | 8.43% | 7ece6d6d-334f-4c1d-830a-66e8773232b7 |
| 210 | 113 | 8514 | 743 | 8.73% | 2deda3d2-6ffc-41f1-8b86-fec4cbf07a5f |
| 163 | 113 | 8508 | 763 | 8.97% | a59c39ef-91a7-4981-8d70-660f9eafd523 |
| 224 | 113 | 8502 | 766 | 9.01% | dd9c18eb-0c91-44f3-99f8-029b24b3bd5f |
| 178 | 113 | 8554 | 784 | 9.17% | 6b044761-0ad3-4da5-bd19-af8fd5d3266f |
| 226 | 15 | 9898 | 944 | 9.54% | 5d910e50-499a-4c7d-8e4e-e1fae56844af |
| 89 | 84 | 8670 | 835 | 9.63% | 0a35d023-9524-4055-a010-44a549a2fc4d |
| 4 | 2 | 10744 | 1045 | 9.73% | 19eb81ce-5c4e-4a80-a552-25c3e74d7d77 |
| 76 | 76 | 11028 | 1077 | 9.77% | bacbae2e-a494-4267-8e6a-bcb37c7f070c |
