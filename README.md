# Let’s do the time warp: diffeomorphic analysis of human white matter pathways over the first year of life

This project examines how infant white matter pathways change shape over time, testing the hypothesis that local expansion is greatest near tract endpoints and lowest in tract cores. Using paired V02 (0–2 months) and V03 (3–9 months) scans from six HBCD infants, complementary anatomical and tract-mask registration pipelines measure along-tract expansion and contraction and assess agreement between the resulting tract deformations.

## Scripts

| Script | Role |
|---|---|
| `dsi_recon_diff_morph.py` | DSI Studio GQI reconstruction |
| `T2w_diff_morph_preproc.sh` | T2w image preprocessing |
| `T2w_diff_morph_4.py` | Anatomical registration and along-tract morphometry |
| `tract2tract_diff_morph_4.py` | Tract-mask registration and along-tract morphometry |
| `run_batch_diff_morph.sh` | Batch runner for both registration pipelines |
| `plot_logj_figures_8.py` | Group analyses and figures |
| `make_niivue_figures.py` | Interactive 3D figures |
