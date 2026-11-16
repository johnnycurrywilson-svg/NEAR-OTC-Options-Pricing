# Before publishing to GitHub

1. Work on this extracted **copy**, never your original Windows model folder.
2. Copy only your verified 12-Nov-2026 `outputs/source_peer_option_nodes.csv` into `outputs/` for LOCAL testing; decide whether redistribution of that data is permitted before committing it. `outputs/` is gitignored.
3. Create a clean virtual environment, install `requirements.txt` and run `pytest`.
4. Run the SVI calibration and diagnostics on the dated raw chain; compare curves, Greeks, premium and reporting metrics with the final PDF.
5. Manually inspect for secrets, access tokens, absolute local paths, personal details, third-party licensing and copyrighted data.
6. Ask an independent reviewer to check data timestamp/quality, SVI fitting, static arbitrage, Greeks, TCA and risk assumptions.
7. Publish to a **private** GitHub repository first; make public only after these checks pass.
8. If raw source data cannot be redistributed, publish permitted minimal derived examples and state that complete replication needs the dated option-chain export.
