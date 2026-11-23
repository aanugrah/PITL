#!/usr/bin/env python3
"""
sample_unflagged_provisions.py

Fully reproducible jurisdiction-stratified random sampling of 100 unflagged provisions
from the 1,053-provision baseline corpus (data/provisions_v2_1.jsonl).

Methodology:
1. Reconstructs the 75-candidate screening funnel from empirical candidate files:
   - 43 distinct lexical keyword matches (Component C1 and C2)
   - 39 distinct LegalBERT semantic retrievals (global top-5 across 8 anchors and nearest per jurisdiction)
   - 7 overlapping provisions identified by both lexical and embedding search
   - Net total: 43 + 39 - 7 = 75 distinct screened candidate provisions.
2. Filters out the 75 flagged candidates from the 1,053 total provisions, yielding exactly
   978 unflagged provisions across the 7 jurisdictions.
3. Implements bounded proportional stratified random sampling with fixed seed 20261108:
   - United Kingdom: 40 (40.0% of sample; N = 417 unflagged)
   - European Union: 26 (26.0% of sample; N = 266 unflagged)
   - China: 13 (13.0% of sample; N = 132 unflagged)
   - Indonesia: 11 (11.0% of sample; N = 118 unflagged; serves as internal control)
   - United States (State): 4 (4.0% of sample; N = 25 unflagged)
   - United States (Federal): 3 (3.0% of sample; N = 12 unflagged)
   - Australia: 3 (3.0% of sample; N = 8 unflagged)
   - Total = exactly 100 provisions.
4. Generates blind coding sheets for author verification:
   - UNFLAGGED_SAMPLE_BLIND_SHEET.xlsx
   - UNFLAGGED_SAMPLE_BLIND_SHEET.csv
   - UNFLAGGED_SAMPLE_BLIND_SHEET.md
   Contains sample_id (UF-001 to UF-100), jurisdiction, doc_id, provision_id, heading, text,
   and blank columns for Author_Code (P/G/F) and Author_Notes.

Author: Ananda Anugrah
Date: 2026-11-08
Seed: 20261108
"""

import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SEED = 20261108

QUOTAS = {
    "United Kingdom": 40,
    "European Union": 26,
    "China": 13,
    "Indonesia": 11,
    "United States (state)": 4,
    "United States (federal)": 3,
    "Australia": 3,
}


def load_corpus_and_candidates(data_dir: Path):
    """Load the full 1,053 provisions and the 75 flagged candidates."""
    prov_file = data_dir / "provisions_v2_1.jsonl"
    if not prov_file.exists():
        raise FileNotFoundError(f"Missing corpus file: {prov_file}")

    with open(prov_file, "r", encoding="utf-8") as f:
        provisions = [json.loads(line) for line in f]
    df_prov = pd.DataFrame(provisions)

    lx_file = data_dir / "lexical_flags_coded.csv"
    t5_file = data_dir / "top5_retrievals_coded.csv"
    t6_file = data_dir / "nearest_by_jurisdiction_coded.csv"

    lx = pd.read_csv(lx_file)
    t5 = pd.read_csv(t5_file)
    t6 = pd.read_csv(t6_file)

    lx_pairs = set(zip(lx["doc_id"], lx["locator"]))
    t5_pairs = set(zip(t5["doc"], t5["locator"]))
    t6_pairs = set(zip(t6["doc"], t6["locator"]))
    ret_pairs = t5_pairs.union(t6_pairs)
    flagged_pairs = lx_pairs.union(ret_pairs)
    overlap_pairs = lx_pairs.intersection(ret_pairs)

    print("=" * 70)
    print("SCREENING FUNNEL RECONCILIATION CHECK")
    print("=" * 70)
    print(f"Total provisions in corpus:             {len(df_prov):>5}")
    print(f"Distinct lexical flags (C1 + C2):       {len(lx_pairs):>5} (Expected: 43)")
    print(f"Distinct LegalBERT retrievals (t5 + t6):{len(ret_pairs):>5} (Expected: 39)")
    print(f"Distinct overlapping candidates:        {len(overlap_pairs):>5} (Expected: 7)")
    print(f"Total distinct flagged candidates:      {len(flagged_pairs):>5} (Expected: 75)")
    assert len(lx_pairs) == 43, f"Expected 43 lexical flags, got {len(lx_pairs)}"
    assert len(ret_pairs) == 39, f"Expected 39 LegalBERT retrievals, got {len(ret_pairs)}"
    assert len(overlap_pairs) == 7, f"Expected 7 overlapping provisions, got {len(overlap_pairs)}"
    assert len(flagged_pairs) == 75, f"Expected 75 distinct candidates, got {len(flagged_pairs)}"

    # Mark flagged provisions
    df_prov["doc_locator_pair"] = list(zip(df_prov["doc_id"], df_prov["locator"]))
    df_prov["is_flagged"] = df_prov["doc_locator_pair"].isin(flagged_pairs)

    n_flagged = df_prov["is_flagged"].sum()
    n_unflagged = (~df_prov["is_flagged"]).sum()
    print(f"Flagged candidates matched in corpus:   {n_flagged:>5} (Expected: 75)")
    print(f"Total unflagged population:             {n_unflagged:>5} (Expected: 978)")
    assert n_flagged == 75, f"Expected 75 flagged in corpus, found {n_flagged}"
    assert n_unflagged == 978, f"Expected 978 unflagged in corpus, found {n_unflagged}"
    print("Funnel arithmetic verified.\n")

    return df_prov, flagged_pairs


def sample_unflagged(df_prov: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Implements bounded proportional stratified random sampling across 7 jurisdictions.
    Deterministically sorts unflagged provisions by provision_id before drawing samples.
    """
    unflagged = df_prov[~df_prov["is_flagged"]].copy()
    unflagged = unflagged.sort_values("provision_id").reset_index(drop=True)

    print("=" * 70)
    print("JURISDICTION-STRATIFIED RANDOM SAMPLING (Seed = 20261108)")
    print("=" * 70)
    print(f"{'Jurisdiction':<26} | {'Unflagged N':<11} | {'Sample n':<8} | {'Sampling %':<10}")
    print("-" * 70)

    rng = np.random.RandomState(seed)
    sampled_dfs = []

    for jur, quota in QUOTAS.items():
        sub = unflagged[unflagged["jurisdiction"] == jur].copy()
        n_available = len(sub)
        assert n_available >= quota, f"Not enough unflagged provisions in {jur}: {n_available} < {quota}"

        chosen_idx = rng.choice(n_available, size=quota, replace=False)
        chosen_idx = sorted(chosen_idx)
        sampled_sub = sub.iloc[chosen_idx].copy()
        sampled_dfs.append(sampled_sub)

        frac = (quota / n_available) * 100
        print(f"{jur:<26} | {n_available:>11} | {quota:>8} | {frac:>9.1f}%")

    sample_concat = pd.concat(sampled_dfs).reset_index(drop=True)
    assert len(sample_concat) == 100, f"Expected exactly 100 sampled provisions, got {len(sample_concat)}"
    print("-" * 70)
    print(f"{'Total Sample':<26} | {len(unflagged):>11} | {len(sample_concat):>8} | {(100/978)*100:>9.1f}%\n")

    # Deterministic permutation for blind sheet presentation
    # This ensures items are not presented in jurisdiction blocks, preventing coder order bias
    perm = rng.permutation(len(sample_concat))
    sample_shuffled = sample_concat.iloc[perm].reset_index(drop=True)

    # Format fields
    sample_shuffled["sample_id"] = [f"UF-{i+1:03d}" for i in range(len(sample_shuffled))]
    sample_shuffled["heading"] = sample_shuffled["locator"]
    sample_shuffled["Author_Code (P/G/F)"] = ""
    sample_shuffled["Author_Notes"] = ""

    return sample_shuffled


def export_blind_sheet_csv(df_sample: pd.DataFrame, out_path: Path):
    """Export blind sample to CSV."""
    cols = [
        "sample_id",
        "jurisdiction",
        "doc_id",
        "provision_id",
        "heading",
        "text",
        "Author_Code (P/G/F)",
        "Author_Notes",
    ]
    df_sample[cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"Exported CSV:   {out_path}")


def export_blind_sheet_excel(df_sample: pd.DataFrame, out_path: Path):
    """Export blind sample to a professional Excel workbook with instructions."""
    wb = openpyxl.Workbook()

    # Sheet 1: Instructions
    ws_inst = wb.active
    ws_inst.title = "Instructions"
    ws_inst.views.sheetView[0].showGridLines = True

    title_font = Font(name="Calibri", size=14, bold=True, color="1F497D")
    sub_font = Font(name="Calibri", size=11, bold=True)
    body_font = Font(name="Calibri", size=11)
    bold_font = Font(name="Calibri", size=11, bold=True)
    code_font = Font(name="Consolas", size=10)

    inst_content = [
        ("INDEPENDENT BLIND CODING SHEET: UNFLAGGED PROVISION CHECK", title_font),
        ("Context & Purpose", sub_font),
        (
            "This sheet contains a jurisdiction-stratified random sample of 100 unflagged provisions drawn from the 978 provisions not flagged during computational screening (seed: 20261108).",
            body_font,
        ),
        (
            "Please review each provision independently against the Parent-in-the-Loop (PITL) co-regulation framework and record your doctrinal evaluation.",
            body_font,
        ),
        ("", body_font),
        ("Coding Categories (Enter in Column G: Author_Code (P/G/F))", sub_font),
        (
            "P = Partial Match: The provision legally mandates or establishes at least one functional core requirement of Component C1 (direct guardian alert on severe interaction risk) OR Component C2 (identity-verified parental supervision tools / authority enforcement / anti-bypass).",
            body_font,
        ),
        (
            "G = General / Adjacent: The provision addresses child safety, age assurance, general parental rights, personal data protection, or child-welfare governance, but lacks an active parental alert trigger or verified supervisory mechanism.",
            body_font,
        ),
        (
            "F = False Positive / True Negative / Unrelated: The provision regulates upstream AI conformity, general commercial obligations, administrative enforcement, institutional governance, general data protection without child focus, or technical standards with no child/parental governance mechanism.",
            body_font,
        ),
        ("", body_font),
        ("Rule of Thumb", sub_font),
        (
            "Ask: 'If this provision were enforced against an AI or digital platform, would it require the provider to send an alert to a verified guardian on serious harm (C1) or enforce parental supervision settings through verified identity / anti-bypass (C2)?' If yes -> P. If it protects children generally or regulates data -> G. If unrelated -> F.",
            body_font,
        ),
        ("", body_font),
        ("Notes Column (Column H)", sub_font),
        (
            "Optional: Record brief doctrinal justification, statutory hook reference, or substantive observation.",
            body_font,
        ),
    ]

    for row_idx, (text, font) in enumerate(inst_content, start=1):
        cell = ws_inst.cell(row=row_idx, column=1, value=text)
        cell.font = font
    ws_inst.column_dimensions["A"].width = 110

    # Sheet 2: Blind Sample
    ws_sample = wb.create_sheet(title="Blind_Sample")
    ws_sample.views.sheetView[0].showGridLines = True

    headers = [
        "sample_id",
        "jurisdiction",
        "doc_id",
        "provision_id",
        "heading",
        "text",
        "Author_Code (P/G/F)",
        "Author_Notes",
    ]

    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    ws_sample.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws_sample.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws_sample.row_dimensions[1].height = 28

    zebra_fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")

    for row_idx, row in df_sample.iterrows():
        excel_row = [
            row["sample_id"],
            row["jurisdiction"],
            row["doc_id"],
            row["provision_id"],
            row["heading"],
            row["text"],
            "",
            "",
        ]
        ws_sample.append(excel_row)
        current_r = row_idx + 2
        ws_sample.row_dimensions[current_r].height = 65

        for col_idx in range(1, len(headers) + 1):
            c = ws_sample.cell(row=current_r, column=col_idx)
            c.border = thin_border
            if row_idx % 2 == 1:
                c.fill = zebra_fill

            if col_idx in [1, 3, 4]:
                c.font = code_font
                c.alignment = Alignment(horizontal="center", vertical="top")
            elif col_idx in [2, 5]:
                c.font = body_font
                c.alignment = Alignment(horizontal="left", vertical="top")
            elif col_idx == 6:
                c.font = body_font
                c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            elif col_idx == 7:
                c.font = Font(name="Calibri", size=12, bold=True, color="C00000")
                c.alignment = Alignment(horizontal="center", vertical="top")
            elif col_idx == 8:
                c.font = body_font
                c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    # Column widths
    ws_sample.column_dimensions["A"].width = 12
    ws_sample.column_dimensions["B"].width = 24
    ws_sample.column_dimensions["C"].width = 12
    ws_sample.column_dimensions["D"].width = 16
    ws_sample.column_dimensions["E"].width = 20
    ws_sample.column_dimensions["F"].width = 75
    ws_sample.column_dimensions["G"].width = 20
    ws_sample.column_dimensions["H"].width = 35

    wb.save(out_path)
    print(f"Exported Excel: {out_path}")


def export_blind_sheet_markdown(df_sample: pd.DataFrame, out_path: Path):
    """Export blind sample to a human-readable Markdown file."""
    md_lines = [
        "# Unflagged Provisions Sample (Blind Coding Sheet)",
        "",
        "**Stratification**: Bounded Proportional Stratified Random Sample (Fixed Seed `20261108`)  ",
        "**Sample Size**: $n = 100$ provisions drawn from $N = 978$ unflagged provisions across 20 instruments.  ",
        "**Coding Rubric**:  ",
        "- `P` = **Partial Match**: Provision mandates guardian alert on severe harm (C1) or verified parental supervision / anti-bypass (C2).",
        "- `G` = **General / Adjacent**: Provision addresses child safety, age assurance, parental rights, or personal data protection without active alert/supervisory mandate.",
        "- `F` = **False Positive / True Negative**: Provision addresses general commercial, administrative, upstream conformity, or institutional matters with no child/parental governance.",
        "",
        "---",
        "",
        "## Summary Inventory of Sampled Provisions",
        "",
        "| Sample ID | Jurisdiction | Doc ID | Provision ID | Heading / Locator | Author Code (P/G/F) | Author Notes |",
        "|---|---|---|---|---|:---:|---|",
    ]

    for _, row in df_sample.iterrows():
        heading_clean = str(row["heading"]).replace("|", "\\|")
        md_lines.append(
            f"| `{row['sample_id']}` | {row['jurisdiction']} | `{row['doc_id']}` | `{row['provision_id']}` | {heading_clean} | | |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## Full Text of Sampled Provisions for Close Doctrinal Reading",
            "",
        ]
    )

    for _, row in df_sample.iterrows():
        md_lines.append(f"### {row['sample_id']}: {row['jurisdiction']} — {row['doc_id']} ({row['heading']})")
        md_lines.append(f"- **Provision ID**: `{row['provision_id']}`")
        md_lines.append(f"- **Document**: `{row['doc_id']}` ({row['jurisdiction']})")
        md_lines.append(f"- **Locator**: {row['heading']}")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(row["text"].strip())
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("- **Author Classification [P / G / F]**: ______")
        md_lines.append("- **Doctrinal Coding Notes**: ")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Exported MD:    {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample unflagged provisions for recall check.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 20261108)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    data_dir = package_root / "data"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = script_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REPRODUCIBLE UNFLAGGED PROVISION SAMPLING SCRIPT")
    print(f"Package Root: {package_root}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Random Seed: {args.seed}")
    print("=" * 70)

    df_prov, flagged_pairs = load_corpus_and_candidates(data_dir)
    df_sample = sample_unflagged(df_prov, seed=args.seed)

    csv_path = out_dir / "UNFLAGGED_SAMPLE_BLIND_SHEET.csv"
    xlsx_path = out_dir / "UNFLAGGED_SAMPLE_BLIND_SHEET.xlsx"
    md_path = out_dir / "UNFLAGGED_SAMPLE_BLIND_SHEET.md"

    export_blind_sheet_csv(df_sample, csv_path)
    export_blind_sheet_excel(df_sample, xlsx_path)
    export_blind_sheet_markdown(df_sample, md_path)

    print("=" * 70)
    print("Sampling and export complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
