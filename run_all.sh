#!/usr/bin/env bash
# Runs the whole analysis for the dissertation in order. Stops at the first error.
set -e
cd "$(dirname "$0")"
mkdir -p results figures

echo "Step 1: check the corpus (needs the LegalBERT tokenizer)";      python3 src/01_verify_corpus.py
echo "Step 2: check anchors and embeddings (needs LegalBERT)";        python3 src/02_verify_anchors_and_embeddings.py
echo "Step 3: topic model and sensitivity";                           python3 src/03_run_thematic_analysis.py
echo "Step 4: LegalBERT diagnostics, keyword rules, retrieval";       python3 src/04_run_pitl_analysis.py
echo "Step 5: OECD incident data";                                    python3 src/05_run_oecd_context.py
echo "Step 6: compare results with the dissertation";                 python3 src/06_generate_report.py
echo "Step 7: figures";                                               python3 src/07_make_figures.py
echo "Done. See results/REPLICATION_REPORT.md and figures/."
