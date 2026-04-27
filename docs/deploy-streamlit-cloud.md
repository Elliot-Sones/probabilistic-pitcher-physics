# Deploying the Pitcher Twin demo to Streamlit Cloud

Streamlit Cloud is free for public repos. The whole flow takes ~10 minutes.

## Prerequisites

- The repo is pushed to GitHub (public or private).
- `data/processed/skubal_2025.csv` is committed (already allowed in `.gitignore`).
- `requirements.txt` is up to date (already verified).

## Steps

1. **Sign in** at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.

2. **Click "Create app"** → "Deploy from GitHub".

3. **Pick the repo** — `trajekt-pitcher-twin` (or whatever you named it on GitHub).

4. **Branch:** `main` &nbsp;·&nbsp; **Main file path:** `app/streamlit_app.py`

5. **App URL:** pick a subdomain like `pitcher-twin` — your URL will become
   `https://pitcher-twin.streamlit.app`.

6. **Advanced settings** (optional):
   - Python version: `3.11`
   - Secrets: none needed for the public Statcast demo

7. **Click "Deploy"** — first build takes ~3-4 minutes (installing scikit-learn, plotly, etc.).

## After it's live

- Update the README's "Live demo" link with your real URL:
  ```markdown
  > **Live demo:** [pitcher-twin.streamlit.app](https://pitcher-twin.streamlit.app)
  ```
- Push the README change. Streamlit auto-redeploys on every push to `main`.

## Resource limits

Free tier gets:
- 1 GB memory
- 1 CPU
- Sleeps after inactivity (wakes in ~10 sec on next visit)

The app loads ~1.9 MB of CSV + small JSON artifacts and fits one suite of
Gaussian-mixture models on first cold visit to the **Try It** tab. That's
well within free-tier limits.

## Troubleshooting

- **"Failed to load app"** on first deploy → check the Streamlit Cloud logs;
  usually a missing dep. Add to `requirements.txt`, push, auto-redeploys.
- **Data path errors** → confirm `data/processed/skubal_2025.csv` is actually
  committed in GitHub (`git ls-files data/processed/`).
- **Slow first load on Try It tab** → expected. The first visit fits the model
  suite; subsequent visits use Streamlit's `@st.cache_resource` decorator.

## Custom domain (optional, later)

You can point a custom domain at the Streamlit Cloud URL if you set up DNS
yourself. Free tier doesn't include managed custom domains.
