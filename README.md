# PACT: Personal-Preference and Cultural-Norm Trade-off

Code and lightweight result artifacts for **Whose Norms? Disentangling Cultural and Personal Alignment in Large Language Models**.

PACT evaluates whether language models choose to follow a cultural norm or allow a personal preference when the two are in tension.

## Repository Layout

```text
scripts/                  Core analysis and plotting scripts
data/model_behavior/       Cleaned benchmark/sample instances for model-behavior analysis
data/prompt_ablation/      Prompt-ablation release data
data/trace_analysis/       Trace-analysis tables and qualitative examples
results/section4/          Model-behavior summary CSVs and figure backing data
results/section5/          Human-study and human-model alignment summary CSVs
figures/section4/          Generated Section 4 figures
figures/section5/          Generated Section 5 figures
```

## Main Scripts

- `scripts/compute_current_eval_metrics.py`: compute model-evaluation summary metrics.
- `scripts/plot_section4_pastel_model_behavior.py`: generate Section 4 model-behavior figures.
- `scripts/plot_section4_demographics_pastel.py`: generate demographic/context plots.
- `scripts/analyze_appendix_model_behavior.py`: appendix analyses for ablations and preference types.
- `scripts/section5_metrics.py`: human-study and human-model alignment metrics.
- `scripts/plot_section5_acl_figures.py`: Section 5 figures.
- `scripts/model_only_section5_analysis.py`: model-only persona/no-persona analyses.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

This release folder intentionally excludes raw scratch outputs, Slurm logs, API keys, and large intermediate model outputs. The included CSVs are lightweight summaries and cleaned release artifacts intended for paper reproduction and inspection.

## Citation

```bibtex
@misc{borah2026pact,
  title={Whose Norms? Disentangling Cultural and Personal Alignment in Large Language Models},
  author={Borah, Angana and Augenstein, Isabelle and Mihalcea, Rada},
  year={2026}
}
```
