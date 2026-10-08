# Confirmation v3 run notes

- Protocol frozen 2026-10-08T10:40:40Z (commit 72d02c8, manifest `MANIFEST_V3.json`); run on the laptop in 9 resumable invocations of `confirmation_v3.py --budget 155`. All 21 manifest files verified on the laptop before the run.
- Results: `confirmation_v3_results.json`, sha256 `727317d7611766d799de9d0f2163a788c7dcfab4635e85ed73c3066ac779b58b`. Plants 7000-7049 are now spent.
- Incident: invocations 3 and 4 were accidentally launched concurrently, so two processes wrote the progress file during plant-conditions ~150-301 (conditions s0.80-s1.10). The simulator is deterministic per plant and every write is a full snapshot, so this can only duplicate work. To verify, every plant in s0.80, s0.90, s1.00 and s1.10 (200 plant-conditions, all controllers and all model initialisations) was recomputed from the frozen code (`verify_overlap.py`). Maximum absolute RMSE difference from the recorded values: **0.0**. The run is unaffected.
