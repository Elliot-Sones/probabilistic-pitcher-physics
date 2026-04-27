# Final Acceptance Checklist

- [ ] Real Statcast cache exists.
- [ ] Real weather cache exists when context/weather mode is enabled.
- [ ] No code path creates mock pitch rows.
- [ ] No code path creates fake weather rows.
- [ ] Test fixtures are real-data slices.
- [ ] Candidate ranking selects the best real pitcher/pitch pair.
- [ ] Same-season validation is complete.
- [ ] Model selected by validation meets or honestly fails the target AUC.
- [ ] Ablation table exists for full contextual mode.
- [ ] Optional Streamlit app runs from real artifacts if built.
- [ ] README shows real metrics and real player results.
