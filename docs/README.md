# PACT Paper Website

Static GitHub Pages site for **Whose Norms? Disentangling Cultural and Personal Alignment in Large Language Models**.

## Public Links

- Paper: https://arxiv.org/pdf/2606.07877
- Code: https://github.com/MichiganNLP/pact_culture_personalization
- Dataset: https://huggingface.co/datasets/Angana192/pact-culture-personalization

## Local Preview

No build step is required. From the repository root, run:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Files

- `index.html`: page structure and public resource links
- `styles.css`: site styling
- `script.js`: interactive charts, tabs, and example carousel
- `data.js`: compact data used by the interactive charts
- `assets/`: static assets retained by the project

## Publish With GitHub Pages

The repository should be configured under **Settings > Pages** with:

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

After editing the site, commit and push the `docs/` directory:

```bash
git status
git add docs
git commit -m "Update paper and dataset links"
git pull --rebase origin main
git push origin main
```

GitHub Pages normally updates within a few minutes. If the old page remains visible, hard-refresh the browser.
