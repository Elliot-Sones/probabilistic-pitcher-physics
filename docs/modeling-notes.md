# Modeling Notes

## GMM Is The First Serious Model

GMM is appropriate for v1 because:

- it is interpretable
- it handles multimodal pitch clouds
- it preserves correlations through full covariance matrices
- it can sample realistic pitch candidates
- BIC provides a defensible component-selection method

## Known Limitations

GMM assumes each cluster is Gaussian. Real pitch distributions may have skew, heavy tails, and hard physical boundaries.

Fallbacks or future models:

- kernel density estimation
- Gaussian copula
- normalizing flow
- conditional VAE
- hierarchical Bayesian model

## Validation Caveat

Do not fit on 2024 and call 2025 AUC pure realism. A pitcher may genuinely change between seasons.

Use:

- same-season temporal holdout for realism
- next-season comparison for drift/generalization

## Plate Location Caveat

`plate_x` and `plate_z` include both pitch shape and command/execution. That is useful for humanized replication, but it must be described honestly.

Use labels:

- shape variability: velocity, spin, release, movement
- command variability: plate location

## Weather Caveat

Do not lead with weather or wind speed.

Reasons:

- public Statcast does not include clean pitch-level wind
- Trajekt sessions are usually cage/indoor contexts
- weather effects are hard to isolate from public data

Human variability is the core signal. External variability can be future work.

