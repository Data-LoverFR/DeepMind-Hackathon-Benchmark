HONESTAI benchmark — quick start

Files in this folder:
- honestai_dataset.json : the 200-item evaluation dataset
- evaluator.py           : evaluation logic (accuracy, ECE, hallucination, abstention, self-correction)
- run_eval.py            : small runner that simulates predictions and runs the evaluator
- requirements.txt       : python deps
- HONESTAI_notebook.ipynb: Kaggle notebook (analysis + visualizations)
- writeup.md             : scientific writeup for the competition
- dataset-metadata.json  : metadata used by the Kaggle CLI (edit `id` before upload)
- sample_submission.jsonl: example submission format (JSONL)
- LICENSE                : CC-BY-4.0 license reference

Quick run (Windows PowerShell):

1) Create a venv (recommended) and install deps:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run the example evaluation (generates `predictions_simulated.jsonl` and `evaluation_summary.json`):

```powershell
python run_eval.py
```

3) To evaluate your own model, produce a `predictions.jsonl` file with one JSON object per line containing:
   { "id": int, "prediction": str, "confidence": float (0..1), "abstain": bool, optional "revised_prediction": str }

Then run:

```powershell
python evaluator.py --dataset honestai_dataset.json --predictions predictions.jsonl
```

Kaggle dataset / competition notes
- The included `dataset-metadata.json` is a starter; update the `id` field to `YOUR_KAGGLE_USERNAME/dataset-slug` before uploading.
- To create a dataset via the Kaggle CLI (after installing `kaggle` and authenticating):

```bash
kaggle datasets create -p . -m dataset-metadata.json
```

- Expected submission format: `sample_submission.jsonl` (one JSON object per line with fields `id`, `prediction`, `confidence`, `abstain`). The evaluator accepts this JSONL format.

License
- This dataset and code are licensed under CC-BY-4.0 (see `LICENSE`). If you prefer a different license for Kaggle, update `dataset-metadata.json` accordingly.

Notebook:
Open `HONESTAI_notebook.ipynb` in Jupyter / Kaggle to load the dataset, visualize calibration and hallucination rates, and compare simulated models.

Contact / notes:
This benchmark is designed to measure metacognitive abilities: calibration, abstention, hallucination detection, and self-correction. Use the evaluate API in `evaluator.py` to compute metrics programmatically.
