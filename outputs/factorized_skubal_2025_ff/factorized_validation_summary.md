# Factorized Physics Validation

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Train rows: `584`
- Holdout rows: `251`

| Layer | Factorized | Game-Drift Gaussian | Game-Drift Copula |
|---|---:|---:|---:|
| command_representation | 0.519 | 0.587 | 0.514 |
| movement_only | 0.554 | 0.588 | 0.507 |
| release_only | 0.575 | 0.641 | 0.612 |
| trajectory_only | 0.600 | 0.629 | 0.650 |
| physics_core | 0.611 | 0.663 | 0.696 |
