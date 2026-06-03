# PACT Paper Website

Static project page for **Whose Norms? Disentangling Cultural and Personal Alignment in Large Language Models**.

## Local preview

Open `index.html` directly in a browser. No build step is required because the compact data are embedded in `data.js`.

If you prefer a local server:

```bash
cd paper_site
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Files

- `index.html`: page structure
- `styles.css`: lilac/lavender/teal theme
- `script.js`: interactive charts, tabs, example carousel
- `data.js`: compact generated data from paper result CSVs
- `assets/`: paper PDF and selected plot images
- `github_hf_release_steps.md`: GitHub and Hugging Face release checklist
- `dataset_card_template.md`: starter Hugging Face dataset card

## Regenerate

The site was generated from local result files in the parent repository. If result CSVs change, rerun the local builder used in this session or rebuild `data.js` from the same CSVs.
