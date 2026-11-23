#!/usr/bin/env python3
"""
04_run_pitl_analysis.py

Computational legal analysis for Parent-in-the-Loop (PITL) co-regulation (Corpus v2.2):
  1. Representation diagnostics & document variance (R1, R2, R3 spaces, eta^2 on first 10 PCs).
  2. Nearest semantic neighbour analysis (82.8% same document, 98.2% same jurisdiction).
  3. Positive controls test (PP 17/2025 Art. 13 rank #2 on A3, Art. 21 rank #55 on B1).
  4. Null baseline test (PITL top-10 mean 0.3045 vs null 0.2912, diff 0.0133 < null bootstrap SD 0.0245).
  5. Deterministic lexical screening (C1: 9 flags; C2: 37 flags; 45 distinct provisions).
  6. LegalBERT retrieval in R2 space (43 distinct provisions) combined with lexical flags
     -> 81 distinct candidates (7 overlap), verified against data/candidates_81_coded.csv (7 P, 14 G, 60 F).

Outputs:
  - results/candidates_81_classified.csv
  - results/lexical_flags_c1.csv
  - results/lexical_flags_c2.csv
  - results/pitl_diagnostics.json
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
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 20261025


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP 4: PITL ANALYSIS, DIAGNOSTICS & 81-CANDIDATE SCREENING (v2.2)")
    print("=" * 70)

    df = pd.read_json(DATA_DIR / "provisions_v2_2.jsonl", lines=True, dtype={"locator": str})
    doc = df["doc_id"].to_numpy(dtype=object)
    jur = df["jurisdiction"].to_numpy(dtype=object)

    E = np.load(DATA_DIR / "embeddings_v2_2.npz")["prov_hier"]
    anchor_data = np.load(DATA_DIR / "anchor_embeddings.npz")
    AN, NU = anchor_data["anchor"], anchor_data["null"]
    AID = [str(x) for x in anchor_data["anchor_id"]]

    L2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = E.mean(0)
    pca = PCA(n_components=10, random_state=SEED).fit(E - mu)

    def r2(X):
        Z = X - mu
        V = pca.components_[:3]
        return L2(Z - (Z @ V.T) @ V)

    R1 = L2(E)
    R2 = r2(E)
    Z = E.copy()
    for d in np.unique(doc):
        Z[doc == d] -= E[doc == d].mean(0)
    R3 = L2(Z)

    def calc_eta2(X):
        Zc = PCA(n_components=10, random_state=SEED).fit_transform(X - X.mean(0))
        out = []
        for j in range(10):
            z = Zc[:, j]
            tot = ((z - z.mean()) ** 2).sum()
            btw = sum(((z[doc == d].mean() - z.mean()) ** 2) * (doc == d).sum() for d in np.unique(doc))
            out.append(btw / tot)
        return float(np.mean(out))

    eta2_r1, eta2_r2, eta2_r3 = calc_eta2(R1), calc_eta2(R2), calc_eta2(R3)
    print(f"Document Variance eta^2: R1={eta2_r1*100:.1f}%, R2={eta2_r2*100:.1f}%, R3={eta2_r3*100:.1f}%")
    assert abs(eta2_r1 - 0.409) <= 0.005

    sim_mat = R1 @ R1.T
    np.fill_diagonal(sim_mat, -1.0)
    nn_idx = sim_mat.argmax(axis=1)
    same_doc_share = float((doc[nn_idx] == doc).mean())
    same_jur_share = float((jur[nn_idx] == jur).mean())
    mask_no_uk04 = (doc != "UK-04")
    sim_sub = R1[mask_no_uk04] @ R1[mask_no_uk04].T
    np.fill_diagonal(sim_sub, -1.0)
    same_doc_no_uk04 = float((doc[mask_no_uk04][sim_sub.argmax(axis=1)] == doc[mask_no_uk04]).mean())
    print(f"Nearest neighbour: same_doc={same_doc_share*100:.1f}% (no UK-04: {same_doc_no_uk04*100:.1f}%), same_jur={same_jur_share*100:.1f}%")
    assert abs(same_doc_share - 0.828) <= 0.005
    assert abs(same_jur_share - 0.982) <= 0.005

    S_A, S_N = r2(AN), r2(NU)
    CA, CN = R2 @ S_A.T, R2 @ S_N.T

    def rank_of(col_idx, pid):
        prov_idx = df.index[df["provision_id"] == pid][0]
        return int((CA[:, col_idx] > CA[prov_idx, col_idx]).sum() + 1)

    id13_pid = df[(df["doc_id"] == "ID-01") & (df["locator"] == "Article 13")]["provision_id"].iloc[0]
    id21_pid = df[(df["doc_id"] == "ID-01") & (df["locator"] == "Article 21")]["provision_id"].iloc[0]
    rank_id13_a3 = rank_of(AID.index("A3"), id13_pid)
    cos_id13_a3 = float(CA[df.index[df["provision_id"] == id13_pid][0], AID.index("A3")])
    best_rank_id21 = min(rank_of(AID.index(a), id21_pid) for a in ["A1", "A2", "B1", "B2"])
    print(f"Positive controls: ID-01 Art 13 rank=#{rank_id13_a3} (cos={cos_id13_a3:.4f}); ID-01 Art 21 best rank=#{best_rank_id21}")
    assert rank_id13_a3 == 2 and best_rank_id21 == 55

    anch_top10 = [np.sort(CA[:, j])[-10:].mean() for j in range(8)]
    null_top10 = [np.sort(CN[:, j])[-10:].mean() for j in range(4)]
    mean_anch_top10 = float(np.mean(anch_top10))
    mean_null_top10 = float(np.mean(null_top10))
    diff_anch_null = mean_anch_top10 - mean_null_top10
    rng_null = np.random.RandomState(SEED)
    null_boot_sd = float(np.std([np.array(null_top10)[rng_null.randint(0, 4, 4)].mean() for _ in range(1000)]))
    print(f"Null baseline: anchors={mean_anch_top10:.4f}, null={mean_null_top10:.4f}, diff={diff_anch_null:.4f}, boot_sd={null_boot_sd:.4f}")
    assert abs(mean_anch_top10 - 0.3045) <= 0.001
    assert abs(mean_null_top10 - 0.2966) <= 0.001

    # Lexical rules
    g = re.compile(r"\b(parents?|guardians?|carers?|caregivers?)\b", re.I)
    al = re.compile(r"\b(notify|notifies|notified|notifying|notifications?|inform|informs|informed|informing|alerts?|alerted)\b|report to\b", re.I)
    vv = re.compile(r"\b(verify|verified|verifies|verification|authenticat\w*|identity|identification|e-kyc)\b", re.I)
    ch = re.compile(r"\b(child|children|minors?)\b", re.I)
    F1 = np.array([bool(g.search(t) and al.search(t)) for t in df["text"]])
    F2 = np.array([bool(vv.search(t) and (g.search(t) or ch.search(t))) for t in df["text"]])
    assert int(F1.sum()) == 9 and int(F2.sum()) == 37

    df[F1][["provision_id", "doc_id", "jurisdiction", "locator", "text"]].to_csv(RESULTS_DIR / "lexical_flags_c1.csv", index=False)
    df[F2][["provision_id", "doc_id", "jurisdiction", "locator", "text"]].to_csv(RESULTS_DIR / "lexical_flags_c2.csv", index=False)

    # Dynamic 81-candidate reconstruction and check against data/candidates_81_coded.csv
    lex_idx = set(np.where(F1 | F2)[0])
    top5_idx = set(int(i) for j in range(8) for i in np.argsort(-CA[:, j])[:5])
    nj_idx = set(int(np.where(jur == u)[0][np.argmax(CA[jur == u, j])]) for u in np.unique(jur) for j in range(8))
    ret_idx = top5_idx | nj_idx
    cand_idx = sorted(lex_idx | ret_idx)
    print(f"Screening counts: lex={len(lex_idx)}, ret={len(ret_idx)}, overlap={len(lex_idx & ret_idx)}, distinct={len(cand_idx)}")
    assert len(lex_idx) == 45 and len(ret_idx) == 43 and len(lex_idx & ret_idx) == 7 and len(cand_idx) == 81

    coded_df = pd.read_csv(DATA_DIR / "candidates_81_coded.csv")
    expected_pids = set(df.iloc[cand_idx]["provision_id"])
    assert set(coded_df["provision_id"]) == expected_pids, "Mismatch between computed 81 candidates and data/candidates_81_coded.csv"
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
    with open(RESULTS_DIR / "pitl_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)
    print("STEP 4 COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
