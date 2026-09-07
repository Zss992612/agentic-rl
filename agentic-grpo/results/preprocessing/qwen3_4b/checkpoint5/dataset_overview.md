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

- Total tokens: 2,638,896
- Min / median / mean / max: 6,149 / 9,911.0 / 10,308.2 / 16,908
- P90 / P95 / P99: 13,531.0 / 13,975.2 / 15,714.8
- Within 32,768: 256
- Over 32,768: 0
- Length bins: `{'04097-08192': 28, '08193-12288': 188, '12289-32768': 40}`

## Supervision

- Total supervised tokens: 415,318
- Total masked tokens: 2,223,578
- Weighted supervised fraction: 15.74%
- Supervised min / median / mean / max: 628 / 1,429.0 / 1,622.3 / 5,999

## Assistant boundary alignment

- Trajectories with merged boundaries: 227
- Merged assistant spans: 343
- Masked leading characters: 686
- Non-whitespace characters masked at boundaries: 0

## Ten longest trajectories

| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |
|---:|---:|---:|---:|---:|---|
| 108 | 99 | 16908 | 5999 | 35.48% | 7cd38137-61b3-4f4e-8da3-3d54bba014cd |
| 107 | 98 | 16901 | 4062 | 24.03% | 056017b6-8da2-4a97-8500-89d59462db5d |
| 106 | 98 | 15843 | 4426 | 27.94% | 9bb35156-39ee-4b1b-8ee5-08bf494ebf80 |
| 201 | 99 | 15610 | 4606 | 29.51% | 110a383a-c797-4a33-8f82-68d80c9b31f8 |
| 244 | 99 | 15473 | 4534 | 29.30% | 5bc962ea-294f-4b16-a1e6-fee2e2c23570 |
| 135 | 98 | 15187 | 3587 | 23.62% | ea435c23-8668-43bc-8cf7-5af21e96fb6f |
| 31 | 20 | 15046 | 3290 | 21.87% | f6b05c85-5d14-4d39-a583-f4bd8da49190 |
| 228 | 99 | 14942 | 3969 | 26.56% | 0126f348-9917-4923-a1af-30f29d57081e |
| 233 | 98 | 14846 | 2101 | 14.15% | 2b1ba4de-cbd5-4bd6-b33d-055da4961b87 |
| 77 | 76 | 14835 | 3516 | 23.70% | 970ea165-9473-433f-807a-d057d40c46dc |

## Ten lowest supervision fractions

| Index | Task | Tokens | Supervised | Fraction | Trajectory ID |
|---:|---:|---:|---:|---:|---|
| 126 | 113 | 8420 | 699 | 8.30% | 52d58da9-e37e-4882-b372-8b3a11e472c1 |
| 191 | 113 | 8431 | 714 | 8.47% | 7ece6d6d-334f-4c1d-830a-66e8773232b7 |
| 210 | 113 | 8518 | 747 | 8.77% | 2deda3d2-6ffc-41f1-8b86-fec4cbf07a5f |
| 163 | 113 | 8512 | 767 | 9.01% | a59c39ef-91a7-4981-8d70-660f9eafd523 |
| 224 | 113 | 8506 | 770 | 9.05% | dd9c18eb-0c91-44f3-99f8-029b24b3bd5f |
| 178 | 113 | 8558 | 788 | 9.21% | 6b044761-0ad3-4da5-bd19-af8fd5d3266f |
| 226 | 15 | 9902 | 948 | 9.57% | 5d910e50-499a-4c7d-8e4e-e1fae56844af |
| 89 | 84 | 8674 | 839 | 9.67% | 0a35d023-9524-4055-a010-44a549a2fc4d |
| 4 | 2 | 10748 | 1049 | 9.76% | 19eb81ce-5c4e-4a80-a552-25c3e74d7d77 |
| 76 | 76 | 11032 | 1081 | 9.80% | bacbae2e-a494-4267-8e6a-bcb37c7f070c |
