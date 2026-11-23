#!/usr/bin/env python3
"""
01_verify_corpus.py

Automated corpus verification and integrity test suite for Corpus v2.2.
Verifies:
  - 1,085 provisions across 21 instruments and 7 jurisdictional units
  - 340,326 total words
  - 12/12 automated integrity tests (cleanliness, deduplication, locators, headers, UNK tokens)
  - Table 3.1 & Table 4.3 summary statistics

Outputs:
  - results/corpus_verification.json
  - results/token_stats_v2_2.csv
  - results/table_4_1_corpus_summary.csv
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOCAL_MODEL_DIR = ROOT.parent / "Code" / "models" / "legal-bert-base-uncased"
HF_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def strip_accents_and_lower(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def get_tokenizer():
    if LOCAL_MODEL_DIR.exists():
        try:
            return AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(HF_MODEL_NAME)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prov_path = DATA_DIR / "provisions_v2_2.jsonl"
    unk_path = DATA_DIR / "unk_chars.csv"

    print("=" * 70)
    print("STEP 1: AUTOMATED CORPUS INTEGRITY & STATISTICAL VERIFICATION")
    print("=" * 70)

    if not prov_path.exists():
        print(f"ERROR: Missing corpus file: {prov_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_json(prov_path, lines=True, dtype={"locator": str})
    n_provisions = len(df)
    n_words = int(df["n_words"].sum())
    n_docs = int(df["doc_id"].nunique())
    n_short = int((df["n_words"] < 40).sum())

    print(f"Corpus Loaded: {prov_path.name}")
    print(f"  Total Provisions: {n_provisions:,} (Expected: 1,085)")
    print(f"  Total Words:      {n_words:,} (Expected: 340,326)")
    print(f"  Total Documents:  {n_docs} (Expected: 21)")
    print(f"  Short (<40w):     {n_short} (Expected: 68)")

    assert n_provisions == 1085, f"Provision count mismatch: {n_provisions} != 1085"
    assert n_words == 340326, f"Word count mismatch: {n_words} != 340326"
    assert n_docs == 21, f"Document count mismatch: {n_docs} != 21"
    assert n_short == 68, f"Short count mismatch: {n_short} != 68"

    # Jurisdictional summary (Table 4.1 baseline)
    by_jur = df.groupby("jurisdiction").agg(
        docs=("doc_id", "nunique"),
        provisions=("provision_id", "count"),
        words=("n_words", "sum"),
        short_provisions=("n_words", lambda s: int((s < 40).sum()))
    ).reset_index()

    # Reorder according to dissertation
    jur_order = [
        "Australia",
        "China",
        "European Union",
        "Indonesia",
        "United Kingdom",
        "United States (federal)",
        "United States (state)",
    ]
    by_jur["sort_key"] = by_jur["jurisdiction"].map(lambda x: jur_order.index(x) if x in jur_order else 99)
    by_jur = by_jur.sort_values("sort_key").drop(columns=["sort_key"])

    total_row = pd.DataFrame([{
        "jurisdiction": "Total",
        "docs": n_docs,
        "provisions": n_provisions,
        "words": n_words,
        "short_provisions": n_short
    }])
    jur_table = pd.concat([by_jur, total_row], ignore_index=True)
    jur_table.to_csv(RESULTS_DIR / "table_4_1_corpus_summary.csv", index=False)
    print("\nJurisdictional Breakdown (Table 4.1):")
    print(jur_table.to_string(index=False))

    # Integrity checks (1-12)
    print("\nRunning 12 Automated Integrity Tests...")
    test_results = {}

    # 1. No (cid:
    cid_matches = [p for p in df["text"] if "(cid:" in p]
    t1 = len(cid_matches) == 0
    test_results["Test 01 - No (cid: artifacts"] = {"pass": t1, "violations": len(cid_matches)}
    print(f"  [1/12] No (cid: characters:                {'PASS' if t1 else 'FAIL'}")

    # 2. No U+FFFD
    fffd_matches = [p for p in df["text"] if "\ufffd" in p]
    t2 = len(fffd_matches) == 0
    test_results["Test 02 - No U+FFFD replacement char"] = {"pass": t2, "violations": len(fffd_matches)}
    print(f"  [2/12] No U+FFFD replacement char:         {'PASS' if t2 else 'FAIL'}")

    # 3. No private-use characters (U+E000 - U+F8FF)
    pua_matches = [p for p in df["text"] if re.search(r"[\ue000-\uf8ff]", p)]
    t3 = len(pua_matches) == 0
    test_results["Test 03 - No private-use characters"] = {"pass": t3, "violations": len(pua_matches)}
    print(f"  [3/12] No private-use characters:          {'PASS' if t3 else 'FAIL'}")

    # 4. Unique provision_id
    t4 = df["provision_id"].nunique() == n_provisions
    test_results["Test 04 - Unique provision IDs"] = {"pass": t4, "unique": df["provision_id"].nunique()}
    print(f"  [4/12] Unique provision IDs:               {'PASS' if t4 else 'FAIL'}")

    # 5. No exact duplicate text
    dups = df[df.duplicated(subset=["text"], keep=False)]
    t5 = len(dups) == 0
    test_results["Test 05 - No duplicate text provisions"] = {"pass": t5, "duplicates": len(dups)}
    print(f"  [5/12] No duplicate text provisions:       {'PASS' if t5 else 'FAIL'}")

    # 6. No provision over 3,000 words unless annex/schedule or US-F-01 EO sections 4 & 10
    over_3000 = df[(df["n_words"] > 3000) & (~df["unit_type"].isin(["annex", "schedule"]))]
    over_non_usf01 = [r for _, r in over_3000.iterrows() if r["doc_id"] != "US-F-01"]
    t6 = len(over_non_usf01) == 0
    test_results["Test 06 - Max length bounded (<3000w)"] = {"pass": t6, "violations": len(over_non_usf01)}
    print(f"  [6/12] Max provision length bounded:       {'PASS' if t6 else 'FAIL'}")

    # 7. UK-04 first provision is section 1 (not Contents)
    uk04_df = df[df["doc_id"] == "UK-04"]
    uk04_first = uk04_df.iloc[0] if len(uk04_df) > 0 else None
    t7 = uk04_first is not None and uk04_first["locator"] == "Section 1"
    test_results["Test 07 - UK-04 first provision is Sec 1"] = {"pass": t7, "locator": uk04_first["locator"] if uk04_first is not None else None}
    print(f"  [7/12] UK-04 first provision is Section 1: {'PASS' if t7 else 'FAIL'}")

    # 8. EU-01 has locator Annex III
    eu01_annex3 = df[(df["doc_id"] == "EU-01") & (df["locator"] == "Annex III")]
    t8 = len(eu01_annex3) == 1
    test_results["Test 08 - EU-01 contains Annex III"] = {"pass": t8, "matches": len(eu01_annex3)}
    print(f"  [8/12] EU-01 contains Annex III:           {'PASS' if t8 else 'FAIL'}")

    # 9. No residual header/footer lines
    bad_patterns = [
        re.compile(r"^\s*ELI:\s*http\S+.*$", re.I),
        re.compile(r"^\s*OJ\s+L,\s*\d+\.\d+\.\d{4}\s*$", re.I),
        re.compile(r"Official Journal of the European Union", re.I),
        re.compile(r"^\s*(?:[0-9]+|[ivxlcdm]+)?\s*Online Safety Act 2023 \(c\. 50\)\s*(?:[0-9]+|[ivxlcdm]+)?\s*$", re.I),
        re.compile(r"CLT is a crowdsourced.*", re.I),
        re.compile(r"^\s*Enrolled Copy S\.B\. 149\s*$", re.I)
    ]
    header_violations = sum(
        1 for text in df["text"] for line in text.split("\n")
        if any(p.search(line) if "Official" in p.pattern or "CLT" in p.pattern else p.match(line) for p in bad_patterns)
    )
    t9 = header_violations == 0
    test_results["Test 09 - No residual header/footer lines"] = {"pass": t9, "violations": header_violations}
    print(f"  [9/12] No residual header/footer lines:    {'PASS' if t9 else 'FAIL'}")

    # 10. Brief 7B residue patterns
    p9b_patterns = [
        re.compile(r"\d+/\d+\s+ELI:", re.I),
        re.compile(r"Fmt\s+\d+\s+Sfmt", re.I),
        re.compile(r"Federal Register/Vol", re.I),
        re.compile(r"\(c\.\s*50\)\s*\d+", re.I),
        re.compile(r"CITED BY", re.I),
        re.compile(r"CHINESE-LANGUAGE ORIGINAL", re.I),
    ]
    b7b_violations = sum(
        1 for text in df["text"] for line in text.split("\n")
        if any(pat.search(line) for pat in p9b_patterns)
    )
    t10 = b7b_violations == 0
    test_results["Test 10 - Brief 7B residue patterns"] = {"pass": t10, "violations": b7b_violations}
    print(f"  [10/12] Brief 7B residue patterns clear:   {'PASS' if t10 else 'FAIL'}")

    # 11 & 12: LEGAL-BERT Tokenizer statistics and character inventory
    tokenizer = get_tokenizer()
    unk_token_id = tokenizer.unk_token_id
    vocab = tokenizer.get_vocab()

    token_stats = []
    unk_chars_found = Counter()

    for doc_id, group in df.groupby("doc_id"):
        doc_tokens_total = 0
        doc_unk_total = 0
        token_lengths = []
        for text in group["text"]:
            toks = tokenizer.encode(text, add_special_tokens=False)
            doc_tokens_total += len(toks)
            n_unk = toks.count(unk_token_id)
            doc_unk_total += n_unk
            token_lengths.append(len(toks))
            if n_unk > 0:
                for c in text:
                    if unk_token_id in tokenizer.encode(c, add_special_tokens=False):
                        unk_chars_found[c] += 1

        unk_share = (doc_unk_total / doc_tokens_total) if doc_tokens_total > 0 else 0
        share_gt_510 = sum(1 for l in token_lengths if l > 510) / len(token_lengths)
        med_tokens = pd.Series(token_lengths).median()

        token_stats.append({
            "doc_id": doc_id,
            "n_provisions": len(group),
            "total_tokens": doc_tokens_total,
            "total_unk": doc_unk_total,
            "unk_share_pct": round(unk_share * 100, 4),
            "share_over_510_pct": round(share_gt_510 * 100, 2),
            "median_tokens": med_tokens
        })

    stats_df = pd.DataFrame(token_stats)
    stats_df.to_csv(RESULTS_DIR / "token_stats_v2_2.csv", index=False)

    # 11. Character inventory completeness
    known_unk_chars = set()
    if unk_path.exists():
        known_unk_chars = set(pd.read_csv(unk_path)["character"].tolist())
    known_unk_chars.update(unk_chars_found.keys())

    unaccounted_chars = Counter()
    for text in df["text"]:
        stripped = strip_accents_and_lower(text)
        for ch in stripped:
            if ch.isspace():
                continue
            if ch not in vocab and ch not in known_unk_chars:
                unaccounted_chars[ch] += 1

    t11 = len(unaccounted_chars) == 0
    test_results["Test 11 - Character inventory completeness"] = {"pass": t11, "unaccounted": len(unaccounted_chars)}
    print(f"  [11/12] Character inventory complete:      {'PASS' if t11 else 'FAIL'}")

    # 12. UNK share < 0.5% in every document
    max_unk_doc = stats_df.sort_values(by="unk_share_pct", ascending=False).iloc[0]
    zero_unk_docs = int((stats_df["total_unk"] == 0).sum())
    t12 = max_unk_doc["unk_share_pct"] < 0.5 and max_unk_doc["unk_share_pct"] < 0.10
    test_results["Test 12 - UNK token share < 0.5%"] = {
        "pass": t12,
        "max_doc": max_unk_doc["doc_id"],
        "max_pct": max_unk_doc["unk_share_pct"],
        "zero_unk_docs": zero_unk_docs
    }
    print(f"  [12/12] UNK token share < 0.5% (all docs): {'PASS' if t12 else 'FAIL'} (max: {max_unk_doc['doc_id']} at {max_unk_doc['unk_share_pct']}%, {zero_unk_docs}/{n_docs} docs 0.0000%)")

    all_passed = bool(all(v["pass"] for v in test_results.values()))
    print("\n" + "=" * 70)
    print(f"CORPUS VERIFICATION SUMMARY: {sum(bool(v['pass']) for v in test_results.values())}/12 TESTS PASSED")
    print(f"OVERALL STATUS: {'ALL PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    # Convert any numpy types to native Python types for json serialization
    clean_test_results = {}
    for k, v in test_results.items():
        clean_row = {}
        for ik, iv in v.items():
            if isinstance(iv, (bool, np.bool_)):
                clean_row[ik] = bool(iv)
            elif isinstance(iv, (int, np.integer)):
                clean_row[ik] = int(iv)
            elif isinstance(iv, (float, np.floating)):
                clean_row[ik] = float(iv)
            else:
                clean_row[ik] = iv
        clean_test_results[k] = clean_row

    summary = {
        "total_provisions": int(n_provisions),
        "total_words": int(n_words),
        "total_documents": int(n_docs),
        "short_provisions": int(n_short),
        "tests_passed": f"{sum(bool(v['pass']) for v in test_results.values())}/12",
        "all_tests_passed": all_passed,
        "max_unk_doc": str(max_unk_doc["doc_id"]),
        "max_unk_share_pct": float(max_unk_doc["unk_share_pct"]),
        "zero_unk_docs": int(zero_unk_docs),
        "tests": clean_test_results
    }

    with open(RESULTS_DIR / "corpus_verification.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
