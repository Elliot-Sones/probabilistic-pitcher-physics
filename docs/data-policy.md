# Data Policy

Pitcher Twin is a real-data project.

Rules:

- Do not create mock pitch rows.
- Do not create synthetic weather rows.
- Do not silently fall back to demo data.
- Tests may use fixture files only if those files are sampled from real public Statcast or real public weather caches.
- Generated model samples must be labeled as simulated outputs.
- If required real data is missing, scripts must fail with a clear fetch/build command.
