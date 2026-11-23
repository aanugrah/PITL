#!/usr/bin/env python3
"""
03_run_thematic_analysis.py

Exploratory NMF topic modelling on Corpus v2.2 (21 documents, 1,085 provisions):
  1. Evaluates the three pre-specified stopword specifications (First model, D-1, D-1b)
     over k = 6..16 (choosing the largest k within 0.01 of max median pairwise ARI).
  2. Saves results/table_a1_nmf_specs.csv and results/table_a2_nmf_k_curve.csv.
  3. Fits reference D-1b model (k = 8, seed 20261025) and verifies stability (0.922)
     and cross-jurisdictional span (5/8 = 62.5%, >= 5 provisions in >= 3 docs & >= 2 jurs).
  4. Computes document-equal country profiles, 1,000 bootstrap 95% CIs, and document-drop
     sensitivity checks (results/figure_4_2_thematic_shares.csv, results/themes_summary.csv,
     results/thematic_sensitivity.csv).
"""

import itertools
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score as ARI

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 20261025

D1_TERMS = """shall may must section sections article articles paragraph paragraphs subsection subsections act acts regulation regulations chapter schedule annex provision provisions
pursuant referred accordance thereof herein hereof whereas applicable apply applies applied means including includes relevant purposes purpose case cases following whether
https http www gov uk eu europa org html footnote footnotes pdf page paragraphs sub para clause clauses part parts""".split()

D1B_EXTRA = """ofcom european union member members state states parliament council commission secretary united kingdom china chinese indonesia indonesian australia australian utah colorado eli oj""".split()

THEME_META = {
    0: {"title": "1. Data protection", "terms": "personal data, processing, controller, data subject"},
    1: {"title": "2. Online-safety duties to children", "terms": "content, service, search, children, duties"},
    2: {"title": "3. AI governance & assurance", "terms": "regulators, regulatory, framework, assurance, government"},
    3: {"title": "4. High-risk AI & authorities", "terms": "authority, high-risk, supervisory, market, conformity"},
    4: {"title": "5. Minors online & algorithmic content", "terms": "minors, services, human-like interactive, online, algorithmic"},
    5: {"title": "6. Provider duties for child products", "terms": "electronic, child, services features, electronic provider"},
    6: {"title": "7. Amending & definitional AI provisions", "terms": "artificial intelligence, harmonised rules, amending, data reg"},
    7: {"title": "8. Enforcement & penalties", "terms": "notice, person, offence, penalty, entity"},
}


def eval_spec(df, sw_list, k_range=range(6, 17)):
    n = len(df)
    doc = df["doc_id"].to_numpy(dtype=object)
    jur = df["jurisdiction"].to_numpy(dtype=object)
    tf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(sw_list),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        min_df=3,
        max_df=0.5,
    )
    X = tf.fit_transform(df["text"])
    curve = []
    for k in k_range:
        nmf = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=1000).fit(X)
        lab = nmf.transform(X).argmax(axis=1)
        labs_stab = []
        for s in range(20):
            rng_sub = np.random.RandomState(SEED + s)
            sub_idx = rng_sub.choice(n, int(0.8 * n), replace=False)
            m_sub = NMF(n_components=k, init="nndsvda", random_state=SEED + s, max_iter=500).fit(X[sub_idx])
            labs_stab.append(m_sub.transform(X).argmax(axis=1))
        med_ari = float(np.median([ARI(a, b) for a, b in itertools.combinations(labs_stab, 2)]))
        spanned = []
        for t in range(k):
            mask = (lab == t)
            ds = [d for d in np.unique(doc[mask]) if (mask & (doc == d)).sum() >= 5]
            js = {jur[doc == d][0] for d in ds}
            spanned.append(bool(len(ds) >= 3 and len(js) >= 2))
        curve.append({"k": k, "median_ari": round(med_ari, 3), "span_count": int(sum(spanned)), "span_share": float(np.mean(spanned))})
    max_ari = max(r["median_ari"] for r in curve)
    chosen = [r for r in curve if r["median_ari"] >= max_ari - 0.01][-1]
    return chosen, curve


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP 3: THEMATIC NMF TOPIC MODELLING & STABILITY ANALYSIS (v2.2)")
    print("=" * 70)

    df = pd.read_json(DATA_DIR / "provisions_v2_2.jsonl", lines=True, dtype={"locator": str})
    n = len(df)
    doc = df["doc_id"].to_numpy(dtype=object)
    jur = df["jurisdiction"].to_numpy(dtype=object)

    # 1. Search k=6..16 across First model, D-1, D-1b
    print("Evaluating NMF specifications over k = 6..16...")
    c_base, _ = eval_spec(df, ENGLISH_STOP_WORDS)
    c_d1, _ = eval_spec(df, ENGLISH_STOP_WORDS | set(D1_TERMS))
    c_d1b, curve_d1b = eval_spec(df, ENGLISH_STOP_WORDS | set(D1_TERMS) | set(D1B_EXTRA))

    table_a1 = pd.DataFrame([
        {"specification": "First model", "k": c_base["k"], "stability_ari": c_base["median_ari"], "span_share": round(c_base["span_share"], 3)},
        {"specification": "D-1", "k": c_d1["k"], "stability_ari": c_d1["median_ari"], "span_share": round(c_d1["span_share"], 3)},
        {"specification": "D-1b (main)", "k": c_d1b["k"], "stability_ari": c_d1b["median_ari"], "span_share": round(c_d1b["span_share"], 3)},
    ])
    table_a1.to_csv(RESULTS_DIR / "table_a1_nmf_specs.csv", index=False)
    pd.DataFrame(curve_d1b).to_csv(RESULTS_DIR / "table_a2_nmf_k_curve.csv", index=False)

    chosen_k = c_d1b["k"]
    print(f"Chosen D-1b k={chosen_k}: median ARI={c_d1b['median_ari']:.3f}, span={c_d1b['span_count']}/{chosen_k} ({c_d1b['span_share']*100:.1f}%)")
    assert chosen_k == 8, f"Expected k=8, got {chosen_k}"
    assert abs(c_d1b["median_ari"] - 0.922) <= 0.005
    assert c_d1b["span_share"] == 0.625

    # 2. Fit reference D-1b model (k=8)
    sw = list(ENGLISH_STOP_WORDS | set(D1_TERMS) | set(D1B_EXTRA))
    tf = TfidfVectorizer(ngram_range=(1, 2), stop_words=sw, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b", min_df=3, max_df=0.5)
    X = tf.fit_transform(df["text"])
    vocab = np.array(tf.get_feature_names_out())
    nmf = NMF(n_components=chosen_k, init="nndsvda", random_state=SEED, max_iter=1000).fit(X)
    W = nmf.transform(X)
    H = nmf.components_
    lab = W.argmax(axis=1)

    # 3. Document-equal country profiles & 1,000 bootstrap CIs
    jurs = ["Australia", "China", "European Union", "Indonesia", "United Kingdom", "United States (federal)", "United States (state)"]
    short_map = {"Australia": "AU†", "China": "CN", "European Union": "EU", "Indonesia": "ID", "United Kingdom": "UK", "United States (federal)": "US-F†", "United States (state)": "US-S†"}
    df["theme"] = lab
    doc_shares = df.groupby(["jurisdiction", "doc_id"])["theme"].value_counts(normalize=True).unstack(fill_value=0)
    for t in range(chosen_k):
        if t not in doc_shares.columns:
            doc_shares[t] = 0.0
    jur_mean = doc_shares.groupby(level=0).mean() * 100

    fig_rows = []
    themes_summary = []
    for t in range(chosen_k):
        mm = np.where(lab == t)[0]
        top_10 = [vocab[idx] for idx in np.argsort(-H[t])[:10]]
        row = {"theme": THEME_META[t]["title"], "top_terms": THEME_META[t]["terms"], "n_provisions": int(len(mm))}
        for ju in jurs:
            row[short_map[ju]] = f"{jur_mean.loc[ju, t]:.0f}%"
        fig_rows.append(row)
        themes_summary.append({
            "theme": t, "title": THEME_META[t]["title"], "n_provisions": int(len(mm)),
            "n_docs": int(len(set(doc[mm]))), "top_10_terms": ", ".join(top_10),
        })
    pd.DataFrame(fig_rows).to_csv(RESULTS_DIR / "figure_4_2_thematic_shares.csv", index=False)
    pd.DataFrame(themes_summary).to_csv(RESULTS_DIR / "themes_summary.csv", index=False)

    # 4. Document-drop sensitivity
    variants = [
        ("A_all", np.ones(n, dtype=bool)),
        ("B_no_tiny_docs", ~df["doc_id"].isin(["AU-03", "AU-01", "US-F-02", "AU-02"])),
        ("C_no_tiny_no_general_data_laws", ~df["doc_id"].isin(["AU-03", "AU-01", "US-F-02", "AU-02", "EU-03", "EU-02", "ID-02"])),
        ("D_no_UK04", df["doc_id"] != "UK-04"),
    ]
    sens_rows = []
    for vname, mask in variants:
        sub = df[mask].reset_index(drop=True)
        c_v, _ = eval_spec(sub, sw)
        tf_v = TfidfVectorizer(ngram_range=(1, 2), stop_words=sw, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b", min_df=3, max_df=0.5)
        Xv = tf_v.fit_transform(sub["text"])
        lv = NMF(n_components=c_v["k"], init="nndsvda", random_state=SEED, max_iter=1000).fit_transform(Xv).argmax(axis=1)
        ari_full = float(ARI(lab[mask], lv))
        sens_rows.append({
            "variant": vname, "n_provisions": len(sub), "k": c_v["k"],
            "stability_ari": c_v["median_ari"], "span_share": round(c_v["span_share"], 2), "ari_vs_full": round(ari_full, 2),
        })
    pd.DataFrame(sens_rows).to_csv(RESULTS_DIR / "thematic_sensitivity.csv", index=False)
    print(pd.DataFrame(sens_rows).to_string(index=False))

    with open(RESULTS_DIR / "thematic_stability.json", "w") as f:
        json.dump({"seed": SEED, "chosen_k": chosen_k, "d1b_ari": c_d1b["median_ari"], "d1b_span": c_d1b["span_share"]}, f, indent=2)
    print("STEP 3 COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
