# arXiv submission checklist (manuscript v4)

**Package:** `manuscript/arxiv_submission_v4.tar.gz`. It contains `main.tex` and four PDF figures, uses only standard TeX Live packages, and the bibliography is inline (no .bib or .bbl needed). It compiles cleanly with two `pdflatex` runs.

**Rebuild from the frozen results:** `python3 make_figures.py && python3 build_tex.py && latexmk -pdf main.tex`. Edit `main_template.tex`, not `main.tex`.

## Before uploading (author's steps)
- [ ] Read the PDF end to end.
- [ ] Optionally get an independent read (roadmap 3.4).
- [ ] Make the GitHub repo public, or upload a Zenodo snapshot, so the availability statement resolves. Then add the DOI to "Code and data availability".
- [ ] Confirm the author line and affiliation are how you want them. Add an email address if you want one.
- [ ] arXiv metadata:
  - primary category cs.LG (you are endorsed there);
  - suggested cross-list eess.SY;
  - title and abstract copied from `main.tex` (the arXiv abstract field takes plain text, so replace `$s$` with s and `\Hz` with Hz);
  - licence of your choice (CC BY 4.0 is common).
- [ ] Comments field, e.g. "12 pages, 4 figures. Synthetic study; code and frozen protocols at <repo URL>".

## Reference status
All references were checked against publisher records, Crossref or arXiv, except:
- **Nagabandi et al. 2018:** cited as ICRA 2018 with its arXiv ID. The IEEE DOI was not confirmed, so it is omitted.
- **Fehrman & Meliza 2023:** year taken from the arXiv ID (2312).
