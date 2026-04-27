# Weather Residual Validation

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Train rows: `584`
- Holdout rows: `251`
- Weather rows used for fit: `584`

| Layer | V2.1 Baseline Mean | V2.2 Weather Mean | Delta |
|---|---:|---:|---:|
| command_representation | 0.532 | 0.541 | +0.010 |
| movement_only | 0.537 | 0.596 | +0.059 |
| release_only | 0.594 | 0.584 | -0.010 |
| trajectory_only | 0.619 | 0.614 | -0.006 |
| physics_core | 0.646 | 0.645 | -0.001 |
