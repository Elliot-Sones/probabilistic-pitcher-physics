# No Mock Data Rules

The overnight agent must not use mock rows, fake players, fake weather, or fabricated examples.

## Allowed

- Real public Statcast rows.
- Real public weather/stadium/context rows.
- Small fixtures sampled from real public caches with provenance metadata.
- Model-generated pitch samples labeled as `simulated_from_real_model`.

## Forbidden

- Mock pitch rows.
- Synthetic weather rows.
- Fake player examples.
- Silent fallback demo data.
- Validation on the same data used to fit the generator.
- Reporting a result without the data path, date range, row count, and validation split.

## Missing Data Rule

If real data is missing, fail loudly and write the fetch command. Do not invent data.

