#!/usr/bin/env python3
"""
04_run_pitl_analysis.py — Representation diagnostics, positive/null controls,
lexical screening, and 81-candidate LegalBERT retrieval verification (PITL Corpus).
"""

import json
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
SEED = 20261025


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 4: PITL ANALYSIS, DIAGNOSTICS & 81-CANDIDATE SCREENING")

    df = pd.read_json(DATA_DIR / "provisions.jsonl", lines=True, dtype={"locator": str})
    doc, jur = df["doc_id"].to_numpy(dtype=object), df["jurisdiction"].to_numpy(dtype=object)
    E = np.load(DATA_DIR / "embeddings.npz")["prov_hier"]
    ad = np.load(DATA_DIR / "anchor_embeddings.npz")
    AN, NU, AID = ad["anchor"], ad["null"], [str(x) for x in ad["anchor_id"]]

    l2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = E.mean(0)
    V = PCA(n_components=10, random_state=SEED).fit(E - mu).components_[:3]
    r2 = lambda X: l2((X - mu) - ((X - mu) @ V.T) @ V)

    R1, R2, Z = l2(E), r2(E), E.copy()
    for d in np.unique(doc):
        Z[doc == d] -= E[doc == d].mean(0)
    R3 = l2(Z)

    def calc_eta2(X):
        Zc = PCA(n_components=10, random_state=SEED).fit_transform(X - X.mean(0))
        return float(np.mean([
            sum(((Zc[doc == d, j].mean() - Zc[:, j].mean()) ** 2) * (doc == d).sum() for d in np.unique(doc))
            / ((Zc[:, j] - Zc[:, j].mean()) ** 2).sum()
            for j in range(10)
        ]))

    eta2_r1, eta2_r2, eta2_r3 = calc_eta2(R1), calc_eta2(R2), calc_eta2(R3)
    print(f"Document Variance eta^2: R1={eta2_r1*100:.1f}%, R2={eta2_r2*100:.1f}%, R3={eta2_r3*100:.1f}%")
    assert abs(eta2_r1 - 0.409) <= 0.005

    sim_mat = R1 @ R1.T
    np.fill_diagonal(sim_mat, -1.0)
    nn_idx = sim_mat.argmax(axis=1)
    same_doc_share, same_jur_share = float((doc[nn_idx] == doc).mean()), float((jur[nn_idx] == jur).mean())
    mask = doc != "UK-04"
    sim_sub = R1[mask] @ R1[mask].T
    np.fill_diagonal(sim_sub, -1.0)
    same_doc_no_uk04 = float((doc[mask][sim_sub.argmax(axis=1)] == doc[mask]).mean())
    print(f"Nearest neighbour: same_doc={same_doc_share*100:.1f}% (no UK-04: {same_doc_no_uk04*100:.1f}%), same_jur={same_jur_share*100:.1f}%")
    assert abs(same_doc_share - 0.828) <= 0.005 and abs(same_jur_share - 0.982) <= 0.005

    CA, CN = R2 @ r2(AN).T, R2 @ r2(NU).T

    def rank_of(col_idx, pid):
        prov_idx = df.index[df["provision_id"] == pid][0]
        return int((CA[:, col_idx] > CA[prov_idx, col_idx]).sum() + 1)

    id13_pid = df[(df["doc_id"] == "ID-01") & (df["locator"] == "Article 13")]["provision_id"].iloc[0]
    id21_pid = df[(df["doc_id"] == "ID-01") & (df["locator"] == "Article 21")]["provision_id"].iloc[0]
    rank_id13_a3 = rank_of(AID.index("A3"), id13_pid)
    cos_id13_a3 = float(CA[df.index[df["provision_id"] == id13_pid][0], AID.index("A3")])
    best_rank_id21 = min(rank_of(AID.index(a), id21_pid) for a in ["A1", "A2", "B1", "B2"])
    print(f"Positive controls: ID-01 Art 13 rank=#{rank_id13_a3} (cos={cos_id13_a3:.4f}); Art 21 best rank=#{best_rank_id21}")
    assert rank_id13_a3 == 2 and best_rank_id21 == 55

    anch_top10 = [np.sort(CA[:, j])[-10:].mean() for j in range(8)]
    null_top10 = [np.sort(CN[:, j])[-10:].mean() for j in range(4)]
    mean_anch_top10, mean_null_top10 = float(np.mean(anch_top10)), float(np.mean(null_top10))
    diff_anch_null = mean_anch_top10 - mean_null_top10
    rng_null = np.random.RandomState(SEED)
    null_boot_sd = float(np.std([np.array(null_top10)[rng_null.randint(0, 4, 4)].mean() for _ in range(1000)]))
    print(f"Null baseline: anchors={mean_anch_top10:.4f}, null={mean_null_top10:.4f}, diff={diff_anch_null:.4f}, boot_sd={null_boot_sd:.4f}")
    assert abs(mean_anch_top10 - 0.3045) <= 0.001 and abs(mean_null_top10 - 0.2966) <= 0.001

    # Lexical rules
    g = re.compile(r"\b(parents?|guardians?|carers?|caregivers?)\b", re.I)
    al = re.compile(r"\b(notify|notifies|notified|notifying|notifications?|inform|informs|informed|informing|alerts?|alerted)\b|report to\b", re.I)
    vv = re.compile(r"\b(verify|verified|verifies|verification|authenticat\w*|identity|identification|e-kyc)\b", re.I)
    ch = re.compile(r"\b(child|children|minors?)\b", re.I)
    F1 = np.array([bool(g.search(t) and al.search(t)) for t in df["text"]])
    F2 = np.array([bool(vv.search(t) and (g.search(t) or ch.search(t))) for t in df["text"]])
    assert int(F1.sum()) == 9 and int(F2.sum()) == 37

    cols = ["provision_id", "doc_id", "jurisdiction", "locator", "text"]
    df[F1][cols].to_csv(RESULTS_DIR / "lexical_flags_c1.csv", index=False)
    df[F2][cols].to_csv(RESULTS_DIR / "lexical_flags_c2.csv", index=False)

    # Dynamic 81-candidate reconstruction
    lex_idx = set(np.where(F1 | F2)[0])
    top5_idx = {int(i) for j in range(8) for i in np.argsort(-CA[:, j])[:5]}
    nj_idx = {int(np.where(jur == u)[0][np.argmax(CA[jur == u, j])]) for u in np.unique(jur) for j in range(8)}
    ret_idx = top5_idx | nj_idx
    cand_idx = sorted(lex_idx | ret_idx)
    print(f"Screening counts: lex={len(lex_idx)}, ret={len(ret_idx)}, overlap={len(lex_idx & ret_idx)}, distinct={len(cand_idx)}")
    assert len(lex_idx) == 45 and len(ret_idx) == 43 and len(lex_idx & ret_idx) == 7 and len(cand_idx) == 81

    coded_df = pd.read_csv(DATA_DIR / "candidates_81_coded.csv")
    assert set(coded_df["provision_id"]) == set(df.iloc[cand_idx]["provision_id"])
    counts = coded_df["class"].value_counts().to_dict()
    assert counts.get("P", 0) == 7 and counts.get("G", 0) == 14 and counts.get("F", 0) == 60
    coded_df.to_csv(RESULTS_DIR / "candidates_81_classified.csv", index=False)

    diag_summary = {
        "corpus": {"n_docs": int(df["doc_id"].nunique()), "n_provisions": len(df), "n_words": int(df["n_words"].sum()), "n_short": int(df["short_flag"].sum())},
        "eta2": {"R1": eta2_r1, "R2": eta2_r2, "R3": eta2_r3},
        "nearest_neighbour": {"same_document_share": same_doc_share, "same_document_no_uk04": same_doc_no_uk04, "same_jurisdiction_share": same_jur_share},
        "positive_controls": {"ID01_Art13_rank_A3": rank_id13_a3, "ID01_Art13_cos_A3": cos_id13_a3, "ID01_Art21_best_rank": best_rank_id21},
        "null_baseline": {"anchor_top10_mean": mean_anch_top10, "null_top10_mean": mean_null_top10, "diff": diff_anch_null, "null_boot_sd": null_boot_sd},
        "screening": {"C1_flags": int(F1.sum()), "C2_flags": int(F2.sum()), "lex_distinct": len(lex_idx), "ret_distinct": len(ret_idx), "overlap": len(lex_idx & ret_idx), "total_distinct": len(cand_idx), "classes": counts},
    }
    (RESULTS_DIR / "pitl_diagnostics.json").write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")
    print("STEP 4 COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
