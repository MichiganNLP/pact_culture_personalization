# GitHub + Hugging Face Release Steps

## 1. Create a clean GitHub repository

```bash
mkdir pact-release
cd pact-release
git init
mkdir -p docs data scripts
cp -r /home/anganab/culture_personalization/paper_site/* docs/
```

Recommended structure:

```text
pact-release/
  docs/                  # GitHub Pages website
  data/                  # small metadata/examples only, not huge raw runs
  scripts/               # clean reproducible scripts
  README.md
  LICENSE
  .gitignore
```

Add a `.gitignore`:

```gitignore
__pycache__/
*.pyc
.DS_Store
.env
*.log
slurm-*.out
raw_outputs/
scratch/
```

Commit and push:

```bash
git add README.md LICENSE .gitignore docs scripts data
git commit -m "Initial PACT release"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

In GitHub, enable Pages from `main` / `docs`.

## 2. Clean the code release

Keep only scripts needed to reproduce public tables/figures:

```text
scripts/
  build_model_behavior_tables.py
  build_human_study_tables.py
  plot_paper_figures.py
  validate_preferences.py
```

Avoid committing scratch paths, API keys, raw Slurm logs, or unreleased annotations.

## 3. Prepare Hugging Face data

Create a dataset repo:

```bash
huggingface-cli login
huggingface-cli repo create pact-culture-personalization --type dataset
```

Recommended dataset layout:

```text
pact-culture-personalization/
  README.md                 # dataset card
  data/
    pact_instances.jsonl
    model_behavior_summary.csv
    human_study_summary.csv
    human_model_alignment_summary.csv
  scripts/
    load_dataset.py          # optional helper
```

Use JSONL for benchmark instances and CSV for aggregate tables. Include a data dictionary with columns, label meanings, and known limitations.

## 4. Upload to Hugging Face

```bash
git clone https://huggingface.co/datasets/<USER>/pact-culture-personalization
cd pact-culture-personalization
cp -r ../clean_data/* .
git add .
git commit -m "Add PACT benchmark data and summaries"
git push
```

## 5. Link everything

Update the website buttons once links exist:

- Paper/arXiv URL
- GitHub code URL
- Hugging Face dataset URL
- Citation/BibTeX
