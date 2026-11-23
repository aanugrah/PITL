#!/usr/bin/env python3
"""
06_generate_report.py

Generates results/REPLICATION_REPORT.md comparing all v2.2 script outputs
against the values reported in DISSERTATION_AFTER_CHINA_ORDER.md.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def main():
    print("=" * 70)
    print("STEP 6: GENERATING REPLICATION REPORT (v2.2)")
    print("=" * 70)

    diag = json.loads((RESULTS_DIR / "pitl_diagnostics.json").read_text())
    stab = json.loads((RESULTS_DIR / "thematic_stability.json").read_text())
    oecd = pd.read_csv(RESULTS_DIR / "oecd_trends_2022_2025.csv").set_index("year")

    c = diag["corpus"]
    sc = diag["screening"]
    nb = diag["null_baseline"]
    nn = diag["nearest_neighbour"]
    pc = diag["positive_controls"]

    rows = [
        ("Corpus documents", "21", str(c["n_docs"])),
        ("Corpus provisions", "1,085", f"{c['n_provisions']:,}"),
        ("Corpus total words", "340,326", f"{c['n_words']:,}"),
        ("Short provisions (<40 words)", "68", str(c["n_short"])),
        ("Lexical C1 flags (Guardian + Alert)", "9", str(sc["C1_flags"])),
        ("Lexical C2 flags (Verify + Child/Guardian)", "37", str(sc["C2_flags"])),
        ("LegalBERT R1 doc variance (eta^2)", "40.9%", f"{diag['eta2']['R1']*100:.1f}%"),
        ("Nearest neighbour same doc (R1)", "82.8%", f"{nn['same_document_share']*100:.1f}%"),
        ("Nearest neighbour same jurisdiction (R1)", "98.2%", f"{nn['same_jurisdiction_share']*100:.1f}%"),
        ("Positive control PP 17/2025 Art. 13 (A3 rank)", "#2", f"#{pc['ID01_Art13_rank_A3']}"),
        ("Positive control PP 17/2025 Art. 21 (best rank)", "#55", f"#{pc['ID01_Art21_best_rank']}"),
        ("Null test: PITL anchors vs null top-10 mean", "0.3045 vs 0.2966", f"{nb['anchor_top10_mean']:.4f} vs {nb['null_top10_mean']:.4f}"),
        ("NMF D-1b chosen k / stability / span", "8 / 0.922 / 62.5%", f"{stab['chosen_k']} / {stab['d1b_ari']:.3f} / {stab['d1b_span']*100:.1f}%"),
        ("Distinct screening candidates (lex / ret / overlap)", "81 (45 / 43 / 7)", f"{sc['total_distinct']} ({sc['lex_distinct']} / {sc['ret_distinct']} / {sc['overlap']})"),
        ("Candidate classification (P / G / F)", "7 / 14 / 60", f"{sc['classes']['P']} / {sc['classes']['G']} / {sc['classes']['F']}"),
        ("OECD 2022 vs 2025 total incidents (6 jurs)", "705 -> 2,749", f"{oecd.loc[2022, 'total_incidents']:,} -> {oecd.loc[2025, 'total_incidents']:,}"),
        ("OECD 2022 vs 2025 child incidents (6 jurs)", "56 -> 305", f"{oecd.loc[2022, 'child_incidents']:,} -> {oecd.loc[2025, 'child_incidents']:,}"),
    ]

    lines = [
        "# Replication Report (Corpus v2.2)",
        "",
        "| Metric | Dissertation (`DISSERTATION_AFTER_CHINA_ORDER.md`) | Reproduced | Match |",
        "|---|---|---|---|",
    ]
    mismatches = 0
    for k, exp, got in rows:
        match = "YES" if exp == got else "NO"
        if match == "NO":
            mismatches += 1
        lines.append(f"| {k} | {exp} | {got} | {match} |")

    out = RESULTS_DIR / "REPLICATION_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out} ({len(rows) - mismatches}/{len(rows)} matched)")
    assert mismatches == 0, f"{mismatches} metric(s) failed to match dissertation values!"
    print("STEP 6 COMPLETE")


if __name__ == "__main__":
    main()
