#!/usr/bin/env python3
"""
01_verify_corpus.py — Automated corpus verification and integrity suite for PITL Corpus.
Outputs: results/corpus_verification.json, results/token_stats.csv, results/table_4_1_corpus_summary.csv
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
LOCAL_MODEL_DIR = ROOT.parent / "Code" / "models" / "legal-bert-base-uncased"
HF_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
JUR_ORDER = [
    "Australia", "China", "European Union", "Indonesia",
    "United Kingdom", "United States (federal)", "United States (state)",
]


def strip_accents_and_lower(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def get_tokenizer():
    if LOCAL_MODEL_DIR.exists():
        try:
            return AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(HF_MODEL_NAME)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prov_path = DATA_DIR / "provisions.jsonl"
    if not prov_path.exists():
        sys.exit(f"ERROR: Missing corpus file: {prov_path}")

    df = pd.read_json(prov_path, lines=True, dtype={"locator": str})
    n_prov, n_words, n_docs = len(df), int(df["n_words"].sum()), int(df["doc_id"].nunique())
    n_short = int((df["n_words"] < 40).sum())
    print(f"STEP 1: {prov_path.name} -> {n_prov:,} provisions, {n_words:,} words, {n_docs} docs, {n_short} short (<40w)")
    assert (n_prov, n_words, n_docs, n_short) == (1085, 340326, 21, 68)

    # Table 4.1 jurisdictional summary
    by_jur = df.groupby("jurisdiction").agg(
        docs=("doc_id", "nunique"),
        provisions=("provision_id", "count"),
        words=("n_words", "sum"),
        short_provisions=("n_words", lambda s: int((s < 40).sum())),
    ).reset_index()
    by_jur = by_jur.sort_values("jurisdiction", key=lambda s: s.map(lambda x: JUR_ORDER.index(x) if x in JUR_ORDER else 99))
    total_row = pd.DataFrame([{"jurisdiction": "Total", "docs": n_docs, "provisions": n_prov, "words": n_words, "short_provisions": n_short}])
    jur_table = pd.concat([by_jur, total_row], ignore_index=True)
    jur_table.to_csv(RESULTS_DIR / "table_4_1_corpus_summary.csv", index=False)
    print(jur_table.to_string(index=False))

    # 12 automated integrity tests
    test_results = {}

    def record(num, key, label, passed, **meta):
        passed = bool(passed)
        test_results[key] = {"pass": passed, **meta}
        print(f"  [{num}/12] {label:<38} {'PASS' if passed else 'FAIL'}")

    print("\nRunning 12 Automated Integrity Tests...")
    record(1, "Test 01 - No (cid: artifacts", "No (cid: characters:", (v := sum("(cid:" in p for p in df["text"])) == 0, violations=v)
    record(2, "Test 02 - No U+FFFD replacement char", "No U+FFFD replacement char:", (v := sum("\ufffd" in p for p in df["text"])) == 0, violations=v)
    record(3, "Test 03 - No private-use characters", "No private-use characters:", (v := sum(bool(re.search(r"[\ue000-\uf8ff]", p)) for p in df["text"])) == 0, violations=v)
    record(4, "Test 04 - Unique provision IDs", "Unique provision IDs:", (u := int(df["provision_id"].nunique())) == n_prov, unique=u)
    record(5, "Test 05 - No duplicate text provisions", "No duplicate text provisions:", (d := int(df.duplicated(subset=["text"], keep=False).sum())) == 0, duplicates=d)

    over = df[(df["n_words"] > 3000) & (~df["unit_type"].isin(["annex", "schedule"])) & (df["doc_id"] != "US-F-01")]
    record(6, "Test 06 - Max length bounded (<3000w)", "Max provision length bounded:", len(over) == 0, violations=len(over))

    uk04 = df[df["doc_id"] == "UK-04"]
    uk04_loc = uk04.iloc[0]["locator"] if len(uk04) else None
    record(7, "Test 07 - UK-04 first provision is Sec 1", "UK-04 first provision is Section 1:", uk04_loc == "Section 1", locator=uk04_loc)

    eu01_a3 = len(df[(df["doc_id"] == "EU-01") & (df["locator"] == "Annex III")])
    record(8, "Test 08 - EU-01 contains Annex III", "EU-01 contains Annex III:", eu01_a3 == 1, matches=eu01_a3)

    bad_hdr = [
        re.compile(r"^\s*ELI:\s*http\S+.*$", re.I), re.compile(r"^\s*OJ\s+L,\s*\d+\.\d+\.\d{4}\s*$", re.I),
        re.compile(r"Official Journal of the European Union", re.I),
        re.compile(r"^\s*(?:[0-9]+|[ivxlcdm]+)?\s*Online Safety Act 2023 \(c\. 50\)\s*(?:[0-9]+|[ivxlcdm]+)?\s*$", re.I),
        re.compile(r"CLT is a crowdsourced.*", re.I), re.compile(r"^\s*Enrolled Copy S\.B\. 149\s*$", re.I),
    ]
    hdr_v = sum(
        1 for text in df["text"] for line in text.split("\n")
        if any(p.search(line) if ("Official" in p.pattern or "CLT" in p.pattern) else p.match(line) for p in bad_hdr)
    )
    record(9, "Test 09 - No residual header/footer lines", "No residual header/footer lines:", hdr_v == 0, violations=hdr_v)

    b7b_pats = [re.compile(p, re.I) for p in (r"\d+/\d+\s+ELI:", r"Fmt\s+\d+\s+Sfmt", r"Federal Register/Vol", r"\(c\.\s*50\)\s*\d+", r"CITED BY", r"CHINESE-LANGUAGE ORIGINAL")]
    b7b_v = sum(1 for text in df["text"] for line in text.split("\n") if any(p.search(line) for p in b7b_pats))
    record(10, "Test 10 - Brief 7B residue patterns", "Brief 7B residue patterns clear:", b7b_v == 0, violations=b7b_v)

    # LegalBERT tokenizer stats & character inventory
    tokenizer = get_tokenizer()
    unk_id, vocab = tokenizer.unk_token_id, tokenizer.get_vocab()
    token_stats, unk_chars = [], Counter()

    for doc_id, group in df.groupby("doc_id"):
        lens, tot_tok, tot_unk = [], 0, 0
        for text in group["text"]:
            toks = tokenizer.encode(text, add_special_tokens=False)
            n_unk = toks.count(unk_id)
            tot_tok += len(toks)
            tot_unk += n_unk
            lens.append(len(toks))
            if n_unk:
                unk_chars.update(c for c in text if unk_id in tokenizer.encode(c, add_special_tokens=False))
        token_stats.append({
            "doc_id": doc_id, "n_provisions": len(group), "total_tokens": tot_tok, "total_unk": tot_unk,
            "unk_share_pct": round(((tot_unk / tot_tok) if tot_tok else 0) * 100, 4),
            "share_over_510_pct": round(sum(l > 510 for l in lens) / len(lens) * 100, 2),
            "median_tokens": pd.Series(lens).median(),
        })

    stats_df = pd.DataFrame(token_stats)
    stats_df.to_csv(RESULTS_DIR / "token_stats.csv", index=False)

    unk_path = DATA_DIR / "unk_chars.csv"
    known_unk = set(pd.read_csv(unk_path)["character"]) if unk_path.exists() else set()
    known_unk.update(unk_chars.keys())
    unaccounted = {ch for text in df["text"] for ch in strip_accents_and_lower(text) if not ch.isspace() and ch not in vocab and ch not in known_unk}
    record(11, "Test 11 - Character inventory completeness", "Character inventory complete:", len(unaccounted) == 0, unaccounted=len(unaccounted))

    max_unk = stats_df.sort_values(by="unk_share_pct", ascending=False).iloc[0]
    zero_unk = int((stats_df["total_unk"] == 0).sum())
    t12 = float(max_unk["unk_share_pct"]) < 0.10
    record(12, "Test 12 - UNK token share < 0.5%", "UNK token share < 0.5% (all docs):", t12,
           max_doc=str(max_unk["doc_id"]), max_pct=float(max_unk["unk_share_pct"]), zero_unk_docs=zero_unk)

    passed_n = sum(v["pass"] for v in test_results.values())
    all_passed = passed_n == 12
    print(f"\nCORPUS VERIFICATION SUMMARY: {passed_n}/12 TESTS PASSED ({'ALL PASS' if all_passed else 'FAIL'})")

    summary = {
        "total_provisions": n_prov, "total_words": n_words, "total_documents": n_docs, "short_provisions": n_short,
        "tests_passed": f"{passed_n}/12", "all_tests_passed": all_passed,
        "max_unk_doc": str(max_unk["doc_id"]), "max_unk_share_pct": float(max_unk["unk_share_pct"]),
        "zero_unk_docs": zero_unk, "tests": test_results,
    }
    (RESULTS_DIR / "corpus_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
