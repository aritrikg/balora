# BALoRA Streamlit research demo

Deterministic and Gaussian Decoder LoRA for Point-Prompted Abdominal CT Segmentation  
Aritrik Ghosh and Sourav Saha · Swami Vivekananda University, Kolkata, India

This app explores saved results from the completed BTCV Fold 1, seed-42 study. The package includes its data and runs on CPU without model downloads or a Kaggle session.

| Page | What you can do |
|---|---|
| Benchmark | Filter 7,398 prediction rows by organ and test volume. Compare Dice, IoU, 2D surface distances and probability metrics. Switch weighting and export the filtered rows. |
| CT cases | Explore 18 anatomy-selected targets. View the matched reference mask, positive point, all three saved masks, segmentation error regions, and Gaussian mutual information. |
| Inference replay | Compare eight saved variants, failure-ranking AUROCs, fallback counts and per-target rates, and warmed encoder timings. Download the underlying tables. |
| Study protocol | Read the data and prompting protocol, view the original method diagram, and inspect the validation curves and checkpoint choice. |

## Run locally

Open a terminal **in the extracted folder that contains `app.py`**. Use Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell use `.venv\Scripts\Activate.ps1` for the activation step. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Extract the ZIP and copy the contents of the `BALoRA_Streamlit` folder to a GitHub repository. Keep `app.py`, `requirements.txt`, `study_data.py`, `data/` and the hidden `.streamlit/` folder together.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, then choose deployment from GitHub.
3. Select the repository and branch and enter `app.py` for the main file path. Choose **Python 3.12** in Advanced settings and select **Deploy**.
4. Open the deployed app, visit the CT case explorer and verify that its images are visible. Test one CSV download before sharing the link.

[Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

For public hosting, include CT-derived assets only where you have permission to redistribute them. If you want a metrics-only repository, omit `data/examples/` and `data/case_index.json`. The benchmark and replay still run and the case explorer explains the available data. Keep the fitted weights and original scans outside the public repository. The app uses no credentials or upload field.

## Results and provenance

- **Primary benchmark.** The completed matched archive has six test volumes, 13 organs, 78 volume–organ pairs and 2,466 shared prompts per method. The three method rows use frozen SAM2, deterministic rank-16 LoRA and Gaussian decoder LoRA with adaptive rank. Pair weighting averages prompts within a volume–organ pair, then gives each pair equal weight. Prompt weighting gives every prompt equal weight.
- **Images.** The 18 saved examples display the archived FP32-thresholded masks and their shared reference mask. Windowed CT was converted to 8-bit grayscale for display, and the Gaussian MI map retains the archive's FP16 values. A zoom crop changes the viewport only. The displayed Dice and IoU refer to the full slice.
- **Replay.** The data come from the verified completed Round 3 inference archive and the subsequent matched analysis of dropout MI beyond organ, area and entropy. Failure-ranking comparisons use fixed labels within each comparison. The retrospective cohort has six volumes and one training seed.
- **External sources.** [BTCV challenge](https://www.synapse.org/Synapse:syn3193805/wiki/89480) and [SAM2 source](https://github.com/facebookresearch/sam2), pinned in the original study to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

The app displays archived results only. It does not collect patients' scans or run new segmentation. Use it for research presentation and exploration.

## Verify or rebuild the bundled data

Run the tests after installing the pinned requirements.

```bash
python -m unittest discover -s tests -v
```

The asset builder reads local artifacts from the completed F1 run and the editable manuscript package. Pass the latter's `analysis/data` directory so it can also load the verified replay tables. The original archive and checkpoints are separate from the demo.

```bash
python scripts/build_assets.py \
  --archive "/path/to/BALoRA_Full_Run_Results(1).zip" \
  --analysis-dir "/path/to/analysis/data"
```

The builder checks the completed archive hash, matches the prompts and shared references, and recomputes Dice from the archived masks. The included `data/method.png` is a raster export of the original paper's method schematic and is supplied separately from the numerical tables.
