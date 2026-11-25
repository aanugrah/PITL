# Code and data for my MSc dissertation

**Beyond the Pacing Problem: A "Parent-in-the-Loop" (PITL) Co-Regulation Framework and Computational Legal Analysis for Adolescent AI Safety**

Ananda Anugrah, MSc Artificial Intelligence and Government, University of Birmingham, 2026.

This repository holds the data and code behind Chapters 1, 3 and 4 of the dissertation (PITL Corpus), so that the numbers, tables and figures can be checked and re-run.

## What the analysis does

1. **PITL Corpus.** 21 AI and child-protection instruments from seven jurisdictional units (US federal, US state, UK, EU, China, Australia, Indonesia), cleaned and split into 1,085 provisions (340,326 words; 68 short provisions under 40 words; Table 3.1, Table 4.3, Appendix A). China Order No. 21 (`CN-05`) was added to the 20-document baseline (1,053 provisions) after initial screening.
2. **Testing LegalBERT.** Provisions and eight PITL "anchor" statements are embedded with LegalBERT. The embeddings mostly reflect the drafting style of each document, and the anchors are not separated from unrelated UK statutory provisions (anchor mean 0.3045 vs null mean 0.2966, difference 0.0079), so in the dissertation the embeddings are used only to retrieve candidate provisions, not to rank countries (Section 4.2, Figure 4.1).
3. **Thematic map.** An exploratory topic model (NMF on TF-IDF, with 60 legal-drafting stopwords plus 23 institutional/jurisdictional proper nouns removed) compares `k = 6..16` and fits `k = 8` (`ARI = 0.922`, cross-jurisdictional span `5/8 = 62.5%`) to map what each jurisdiction's instruments are about (Section 4.3, Figure 4.2, Appendix A, Tables A1–A5).
4. **Screening for PITL.** Keyword rules (45 lexical matches: 9 C1, 37 C2, 1 overlap) and LegalBERT retrieval (43 unique provisions from top-5-per-anchor and nearest-per-jurisdiction) select 81 unique candidate provisions, which I then read and classify as partial (`7 P`), general (`14 G`) or false positive (`60 F`) (Section 4.4, Table 4.4, Figures 3.1, 4.3 and 4.4, Appendix D). **The classes are my own reading**, stored in `data/candidates_81_coded.csv`; no model produces them.
5. **Recall check.** A jurisdiction-stratified random sample of 100 of the 978 unflagged provisions from the 20-document baseline (seed 20261108), coded by the author with the same P/G/F rules; result 0 P, 10 G, 90 F; one-sided 95% upper bound on the share of partial matches 3.3% outside Indonesia (0/89) and 3.0% overall (0/100) (Section 3.3.5, Section 4.4, Appendix E).
6. **OECD context.** Growth in recorded AI incidents (705 in 2022 to 2,749 in 2025) and child-linked incidents (56 to 305) across six jurisdictions (US, UK, EU27, China, Australia, Indonesia) in Section 1.1 (Figure 1.1). These counts are descriptive and say nothing about the effect of any policy.

## How to run

```bash
# clone the repository and open the folder
git clone https://github.com/aanugrah/PITL.git
cd PITL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Steps 1 and 2 download the LegalBERT model (`nlpaueb/legal-bert-base-uncased`) from Hugging Face on the first run. Steps 3 to 7 need only the files in `data/` and run in a few minutes on a laptop. Generated outputs (`results/` and `figures/`) are excluded from the repository and created automatically when you run `bash run_all.sh`; `results/REPLICATION_REPORT.md` compares the reproduced numbers against the dissertation.

You can also regenerate the blind 100-provision recall sample sheet cleanly without overwriting the author's completed workbook:

```bash
python3 recall_check/sample_unflagged_provisions.py
```

By default this writes the uncoded templates to `results/recall_blind_sheet/`.

## Files

| File | What it is |
|---|---|
| `data/provisions.jsonl` | The PITL Corpus: 1,085 provisions across 21 instruments with document, jurisdiction and locator |
| `data/embeddings.npz`, `data/anchor_embeddings.npz` | LegalBERT embeddings of the 1,085 provisions, anchors and null statements |
| `data/PITL_ANCHORS_v1.0.csv` (+ `.sha256`) | The eight anchor statements, frozen before any similarity was computed |
| `data/anchors_null.csv` | Four unrelated UK statutory provisions (tax, companies, harbours, patents) used as a negative control |
| `data/unk_chars.csv` | Known non-vocabulary Unicode characters audited in `01_verify_corpus.py` |
| `data/candidates_81_coded.csv` | Author's P/G/F classification and short reading/reason for all 81 unique PITL Corpus candidate provisions (45 lexical + 43 retrieval, 7 overlap) |
| `data/provisions_v2_1.jsonl`, `data/lexical_flags_coded.csv`, `data/top5_retrievals_coded.csv`, `data/nearest_by_jurisdiction_coded.csv` | Baseline v2.1 inputs for the recall sample only |
| `recall_check/UNFLAGGED_SAMPLE_BLIND_SHEET.xlsx` | Completed recall check: 100 sampled unflagged provisions from the 20-document baseline with the author's codes (`0 P / 10 G / 90 F`) and notes |
| `recall_check/sample_unflagged_provisions.py` | Reproduces the 100-provision stratified recall sample (`seed=20261108`) into `results/recall_blind_sheet/` |
| `data/oecd_aim_children_by_location.csv` | OECD AI Incidents and Hazards Monitor export by location, incidents with children as harmed entities, January 2020 to October 2026, used to select jurisdictions (Section 3.3.1) |
| `data/oecd_ai_incidents_filtered_child.csv`, `data/oecd_eu27_all.csv`, `data/oecd_eu27_child.csv` | Monthly OECD AI incident counts (5 countries + EU27 aggregate) |
| `src/01` to `src/07` | One script per step (see `run_all.sh`) |
| `results/` *(generated by `run_all.sh`)* | Reproduced tables, diagnostics and `REPLICATION_REPORT.md` |
| `figures/` *(generated by `07_make_figures.py`)* | Reproduced empirical figures (Figures 1.1, 4.1, 4.2 and 4.4) |

Where each generated output appears in the dissertation:

| Dissertation | Script | Generated output |
|---|---|---|
| Table 3.1, Table 4.3, Appendix A (corpus) | `01_verify_corpus.py`, `04_run_pitl_analysis.py` | `results/table_4_1_corpus_summary.csv`, `results/token_stats.csv`, `results/lexical_flags_c1.csv`, `results/lexical_flags_c2.csv` |
| Section 4.2, Figure 4.1 | `04_run_pitl_analysis.py`, `07_make_figures.py` | `results/pitl_diagnostics.json`, `figures/fig4_1_legalbert_diagnostic.png` |
| Section 4.3, Figure 4.2, Appendix A (Tables A1–A5) | `03_run_thematic_analysis.py`, `07_make_figures.py` | `results/table_a1_nmf_specs.csv`, `results/table_a2_nmf_k_curve.csv`, `results/figure_4_2_thematic_shares.csv`, `results/themes_summary.csv`, `results/thematic_sensitivity.csv`, `figures/fig4_2_theme_heatmap.png` |
| Section 4.4, Table 4.4, Figure 4.4, Appendix D | `04_run_pitl_analysis.py`, `07_make_figures.py` | `results/candidates_81_classified.csv`, `results/lexical_flags_c1.csv`, `results/lexical_flags_c2.csv`, `figures/fig4_4_pitl_indication_matrix.png` |
| Section 3.3.5, Section 4.4, Appendix E | `recall_check/sample_unflagged_provisions.py` | `recall_check/UNFLAGGED_SAMPLE_BLIND_SHEET.xlsx` *(coded)*; `results/recall_blind_sheet/` *(regenerated blind sheet)* |
| Section 1.1, Figure 1.1 | `05_run_oecd_context.py`, `07_make_figures.py` | `results/oecd_trends_2022_2025.csv`, `results/oecd_yearly_by_jurisdiction.csv`, `figures/fig1_1_oecd_trends.png` |

## Notes on the texts

- UK, EU, US and Australian instruments are official texts.
- Chinese instruments are unofficial English translations from Stanford DigiChina, China Law Translate (Yale Law School Paul Tsai China Center) and CSET (Georgetown). Please refer to those sources for their terms of use.
- The two Indonesian instruments (PP 17/2025 and Law 27/2022) were translated into English with AI assistance and checked by back-translating 20 articles. The Indonesian originals govern.

This repository is shared as supporting material for the dissertation and for academic checking.
