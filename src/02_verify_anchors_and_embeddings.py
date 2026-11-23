#!/usr/bin/env python3
"""
02_verify_anchors_and_embeddings.py

Verifies:
  1. Cryptographic SHA-256 frozen anchor hash against PITL_ANCHORS_v1.0.sha256.
  2. Embeddings dimensional shapes (provisions_hier: 1085x768, anchors: 8x768, null: 4x768).
  3. Re-embedding consistency across 5 sampled provisions (seed 20261029) ensuring max abs difference < 1e-4.
  4. Cosine similarity matrix between PITL anchors (A1-B4) and null controls (N1-N4).

Outputs:
  - results/anchor_cosines.csv
  - results/anchor_verification.json
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
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOCAL_MODEL_DIR = ROOT.parent / "Code" / "models" / "legal-bert-base-uncased"
HF_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_embedder():
    model_src = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_src)
    model = AutoModel.from_pretrained(model_src).eval()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return tokenizer, model, dev, model_src


def embed_text_hierarchical(text: str, tok, model, dev, max_tokens: int = 512):
    ids = tok.encode(text, add_special_tokens=False)
    body = max_tokens - 2
    if len(ids) <= body:
        pieces = [ids]
    else:
        pieces = [ids[i : i + body] for i in range(0, len(ids), body)]

    vecs = []
    lens = []
    for p in pieces:
        seq = [tok.cls_token_id] + p + [tok.sep_token_id]
        inp = torch.tensor([seq], dtype=torch.long, device=dev)
        att = torch.ones_like(inp, device=dev)
        with torch.no_grad():
            out = model(inp, attention_mask=att).last_hidden_state
            # Mean pooling over tokens
            v = out.mean(dim=1).squeeze(0).cpu().numpy()
            vecs.append(v)
            lens.append(len(p))

    vecs = np.array(vecs)
    lens = np.array(lens, dtype=float)
    hier_vec = (vecs * lens[:, None]).sum(0) / lens.sum()
    return hier_vec


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP 2: SHA-256 ANCHOR INTEGRITY & EMBEDDING VERIFICATION")
    print("=" * 70)

    # 1. SHA-256 Check
    anchors_csv_path = DATA_DIR / "PITL_ANCHORS_v1.0.csv"
    anchors_sha_path = DATA_DIR / "PITL_ANCHORS_v1.0.sha256"

    if not anchors_csv_path.exists() or not anchors_sha_path.exists():
        print("ERROR: Missing anchor files in data directory.", file=sys.stderr)
        sys.exit(1)

    calc_hash = compute_sha256(anchors_csv_path)
    stored_hash = anchors_sha_path.read_text(encoding="utf-8").strip().split()[0]

    print(f"Calculated SHA-256: {calc_hash}")
    print(f"Stored SHA-256:     {stored_hash}")
    hash_match = (calc_hash == stored_hash)
    print(f"Anchor Hash Verification: {'PASS (hash matches)' if hash_match else 'FAIL (MISMATCH)'}")
    assert hash_match, f"SHA-256 mismatch: {calc_hash} != {stored_hash}"

    # 2. Embedding Shapes Verification
    emb_path = DATA_DIR / "embeddings_v2_2.npz"
    anchor_emb_path = DATA_DIR / "anchor_embeddings.npz"

    assert emb_path.exists(), f"Missing {emb_path}"
    assert anchor_emb_path.exists(), f"Missing {anchor_emb_path}"

    emb_data = np.load(emb_path)
    anchor_data = np.load(anchor_emb_path)

    prov_hier = emb_data["prov_hier"]
    anchors = anchor_data["anchor"]
    nulls = anchor_data["null"]
    anchor_ids = [str(x) for x in anchor_data["anchor_id"]]
    null_ids = [str(x) for x in anchor_data["null_id"]]

    print(f"\nEmbedding Shapes:")
    print(f"  Provisions (prov_hier): {prov_hier.shape} (Expected: (1085, 768))")
    print(f"  PITL Anchors (A1-B4):   {anchors.shape} (Expected: (8, 768))")
    print(f"  Null Controls (N1-N4):  {nulls.shape} (Expected: (4, 768))")

    assert prov_hier.shape == (1085, 768), f"prov_hier shape {prov_hier.shape} != (1085, 768)"
    assert anchors.shape == (8, 768), f"anchors shape {anchors.shape} != (8, 768)"
    assert nulls.shape == (4, 768), f"nulls shape {nulls.shape} != (4, 768)"
    print("Embedding Shapes Check: PASS")

    # 3. Re-embedding Consistency Check (Sample 5 provisions, seed 20261029)
    print("\nInitializing LegalBERT embedder for consistency check...")
    tok, model, dev, model_src = get_embedder()
    print(f"Embedder source: {model_src} on device: {dev}")

    df_prov = pd.read_json(DATA_DIR / "provisions_v2_2.jsonl", lines=True)
    rng = np.random.default_rng(20261029)
    sample_indices = rng.choice(len(df_prov), size=5, replace=False)

    print(f"Testing 5 sample provisions (seed 20261029): {sample_indices.tolist()}")
    diffs = []
    sample_reports = []

    for idx in sample_indices:
        row = df_prov.iloc[idx]
        text = row["text"]
        stored_vec = prov_hier[idx]
        recalc_vec = embed_text_hierarchical(text, tok, model, dev, max_tokens=512)
        max_diff = float(np.max(np.abs(stored_vec - recalc_vec)))
        diffs.append(max_diff)
        sample_reports.append({
            "index": int(idx),
            "provision_id": str(row["provision_id"]),
            "doc_id": str(row["doc_id"]),
            "locator": str(row["locator"]),
            "max_abs_diff": max_diff
        })
        print(f"  Provision {row['provision_id']} ({row['doc_id']} {row['locator']}): max abs diff = {max_diff:.2e}")

    overall_max_diff = max(diffs)
    consistency_pass = bool(overall_max_diff < 1e-4)
    print(f"Overall Max Re-embedding Difference: {overall_max_diff:.2e}")
    print(f"Consistency Check (<1e-4): {'PASS' if consistency_pass else 'FAIL'}")
    assert consistency_pass, f"Re-embedding difference {overall_max_diff:.2e} exceeded tolerance 1e-4"

    # 4. Cosine Similarity Matrix (8 Anchors x 8 Anchors + 4 Nulls)
    A_norm = anchors / np.clip(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-12, None)
    N_norm = nulls / np.clip(np.linalg.norm(nulls, axis=1, keepdims=True), 1e-12, None)

    sim_AA = A_norm @ A_norm.T
    sim_AN = A_norm @ N_norm.T
    full_sim = np.hstack([sim_AA, sim_AN])
    all_cols = anchor_ids + null_ids

    df_cos = pd.DataFrame(full_sim, index=anchor_ids, columns=all_cols)
    cos_path = RESULTS_DIR / "anchor_cosines.csv"
    df_cos.to_csv(cos_path)
    print(f"\nSaved Cosine Similarity Matrix to {cos_path}")
    print("\nCosine Similarities (PITL Anchors vs Anchors & Nulls):")
    print(df_cos.round(4).to_string())

    # Off-diagonal anchor-anchor cosines
    mask_off_diag = ~np.eye(8, dtype=bool)
    aa_off_diag = sim_AA[mask_off_diag]
    raw_aa_min, raw_aa_max = float(aa_off_diag.min()), float(aa_off_diag.max())
    raw_an_min, raw_an_max = float(sim_AN.min()), float(sim_AN.max())

    print(f"\nRaw Cosine Range between PITL Anchors: [{raw_aa_min:.3f}, {raw_aa_max:.3f}] (Expected ~0.859-0.941)")
    print(f"Raw Cosine Range vs Unrelated Nulls:     [{raw_an_min:.3f}, {raw_an_max:.3f}] (Expected ~0.821-0.897)")

    summary = {
        "anchors_sha256": calc_hash,
        "anchors_sha256_match": hash_match,
        "shapes": {
            "prov_hier": list(prov_hier.shape),
            "anchor": list(anchors.shape),
            "null": list(nulls.shape)
        },
        "consistency_check": {
            "pass": consistency_pass,
            "max_abs_diff": overall_max_diff,
            "samples": sample_reports
        },
        "raw_cosine_ranges": {
            "anchor_anchor_min": raw_aa_min,
            "anchor_anchor_max": raw_aa_max,
            "anchor_null_min": raw_an_min,
            "anchor_null_max": raw_an_max
        }
    }

    with open(RESULTS_DIR / "anchor_verification.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE: ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
