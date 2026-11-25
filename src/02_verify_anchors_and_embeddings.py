#!/usr/bin/env python3
"""
02_verify_anchors_and_embeddings.py — Verifies SHA-256 anchor hash, embedding shapes,
re-embedding consistency (<1e-4), and anchor-vs-null cosine similarity matrix.
"""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
LOCAL_MODEL_DIR = ROOT.parent / "Code" / "models" / "legal-bert-base-uncased"
HF_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def get_embedder():
    src = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_NAME
    tok, model = AutoTokenizer.from_pretrained(src), AutoModel.from_pretrained(src).eval()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model.to(dev), dev, src


def embed_text_hierarchical(text: str, tok, model, dev, max_tokens: int = 512):
    ids = tok.encode(text, add_special_tokens=False)
    body = max_tokens - 2
    pieces = [ids[i : i + body] for i in range(0, max(1, len(ids)), body)]
    vecs, lens = [], []
    with torch.no_grad():
        for p in pieces:
            inp = torch.tensor([[tok.cls_token_id] + p + [tok.sep_token_id]], dtype=torch.long, device=dev)
            v = model(inp, attention_mask=torch.ones_like(inp)).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            vecs.append(v)
            lens.append(len(p))
    lens_arr = np.array(lens, dtype=float)
    return (np.array(vecs) * lens_arr[:, None]).sum(0) / lens_arr.sum()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 2: SHA-256 ANCHOR INTEGRITY & EMBEDDING VERIFICATION")

    anchors_csv, anchors_sha = DATA_DIR / "PITL_ANCHORS_v1.0.csv", DATA_DIR / "PITL_ANCHORS_v1.0.sha256"
    if not anchors_csv.exists() or not anchors_sha.exists():
        sys.exit("ERROR: Missing anchor files in data directory.")

    calc_hash = hashlib.sha256(anchors_csv.read_bytes()).hexdigest()
    stored_hash = anchors_sha.read_text(encoding="utf-8").strip().split()[0]
    hash_match = calc_hash == stored_hash
    print(f"SHA-256 match: {hash_match} ({calc_hash[:16]}...)")
    assert hash_match, f"SHA-256 mismatch: {calc_hash} != {stored_hash}"

    prov_hier = np.load(DATA_DIR / "embeddings.npz")["prov_hier"]
    ad = np.load(DATA_DIR / "anchor_embeddings.npz")
    anchors, nulls = ad["anchor"], ad["null"]
    anchor_ids, null_ids = [str(x) for x in ad["anchor_id"]], [str(x) for x in ad["null_id"]]
    assert prov_hier.shape == (1085, 768) and anchors.shape == (8, 768) and nulls.shape == (4, 768)
    print(f"Shapes PASS: prov={prov_hier.shape}, anchors={anchors.shape}, null={nulls.shape}")

    # Re-embedding consistency check (5 random provisions, seed 20261029)
    tok, model, dev, src = get_embedder()
    print(f"Embedder: {src} ({dev})")
    df_prov = pd.read_json(DATA_DIR / "provisions.jsonl", lines=True)
    sample_idx = np.random.default_rng(20261029).choice(len(df_prov), size=5, replace=False)

    sample_reports = []
    for idx in sample_idx:
        row = df_prov.iloc[idx]
        max_diff = float(np.max(np.abs(prov_hier[idx] - embed_text_hierarchical(row["text"], tok, model, dev))))
        sample_reports.append({"index": int(idx), "provision_id": str(row["provision_id"]), "doc_id": str(row["doc_id"]), "locator": str(row["locator"]), "max_abs_diff": max_diff})
        print(f"  {row['provision_id']} ({row['doc_id']} {row['locator']}): max diff = {max_diff:.2e}")

    overall_max = max(r["max_abs_diff"] for r in sample_reports)
    assert overall_max < 1e-4, f"Re-embedding difference {overall_max:.2e} exceeded 1e-4"
    print(f"Consistency check PASS (max={overall_max:.2e})")

    # Cosine similarity matrix (8 anchors x 8 anchors + 4 nulls)
    l2 = lambda X: X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    A_norm, N_norm = l2(anchors), l2(nulls)
    sim_AA, sim_AN = A_norm @ A_norm.T, A_norm @ N_norm.T
    df_cos = pd.DataFrame(np.hstack([sim_AA, sim_AN]), index=anchor_ids, columns=anchor_ids + null_ids)
    df_cos.to_csv(RESULTS_DIR / "anchor_cosines.csv")

    aa_off = sim_AA[~np.eye(8, dtype=bool)]
    summary = {
        "anchors_sha256": calc_hash,
        "anchors_sha256_match": hash_match,
        "shapes": {"prov_hier": list(prov_hier.shape), "anchor": list(anchors.shape), "null": list(nulls.shape)},
        "consistency_check": {"pass": True, "max_abs_diff": overall_max, "samples": sample_reports},
        "raw_cosine_ranges": {
            "anchor_anchor_min": float(aa_off.min()), "anchor_anchor_max": float(aa_off.max()),
            "anchor_null_min": float(sim_AN.min()), "anchor_null_max": float(sim_AN.max()),
        },
    }
    (RESULTS_DIR / "anchor_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STEP 2 COMPLETE: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
