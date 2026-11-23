#!/usr/bin/env python3
"""
07_make_figures.py

Generates all dissertation figures (Figures 1.1, 3.1, Diagram 4.1, 4.1, 4.2, 4.3, 4.4, 5.1)
directly from the v2.2 data and results into figures/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
SEED = 20261025


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Serif"
    arr = dict(arrowstyle="-|>", lw=1.4, color="#445566", mutation_scale=12)

    # 1. Figure 1.1: OECD Yearly Combo-Axis Trends by Jurisdiction (2020-2025)
    oecd_yr = pd.read_csv(RESULTS_DIR / "oecd_yearly_by_jurisdiction.csv")
    jurs = ["United States", "United Kingdom", "European Union (EU27)", "China", "Australia", "Indonesia"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), dpi=300, sharex=True)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    x = np.arange(len(years))
    h1, l1, h2, l2 = None, None, None, None
    for idx, (ax, jur) in enumerate(zip(axes.flat, jurs)):
        sub = oecd_yr[oecd_yr["jurisdiction"] == jur].set_index("year").reindex(years)
        tot = sub["total_incidents"].fillna(0).astype(int).values
        ch = sub["child_incidents"].fillna(0).astype(int).values
        b = ax.bar(x, tot, width=0.58, color="#4A77A8", alpha=0.85, label="All recorded incidents (left axis)")
        ax.set_title(jur, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_ylim(0, max(10, int(tot.max() * 1.22)))
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5, colors="#2E547A")
        ax2 = ax.twinx()
        ln = ax2.plot(x, ch, color="#C85A32", marker="o", markersize=4.5, lw=1.8,
                      label="Child & youth incidents (right axis)")
        ax2.set_ylim(0, max(5, int(ch.max() * 1.28)))
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", labelsize=7.5, colors="#A0401E")
        if idx % 3 == 0:
            ax.set_ylabel("All incidents", fontsize=8, color="#2E547A")
        if idx % 3 == 2:
            ax2.set_ylabel("Child & youth", fontsize=8, color="#A0401E")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=7.5, rotation=25)
        if h1 is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("OECD AI Incidents and Hazards Monitor by Jurisdiction (2020–2025)", fontsize=10.5, fontweight="bold", y=0.945)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(FIG_DIR / "fig1_1_oecd_trends.png", dpi=300)
    plt.close()


    # 4. Figure 4.1: LegalBERT diagnostic
    E = np.load(DATA_DIR / "embeddings_v2_2.npz")["prov_hier"]
    ad = np.load(DATA_DIR / "anchor_embeddings.npz")
    AN, NU = ad["anchor"], ad["null"]
    L2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = E.mean(0)
    V = PCA(n_components=10, random_state=SEED).fit(E - mu).components_[:3]
    r2 = lambda X: L2((X - mu) - ((X - mu) @ V.T) @ V)
    CA, CN = r2(E) @ r2(AN).T, r2(E) @ r2(NU).T
    anch_t10 = [np.sort(CA[:, j])[-10:].mean() for j in range(8)]
    null_t10 = [np.sort(CN[:, j])[-10:].mean() for j in range(4)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=300)
    ax.scatter(anch_t10, [1] * 8, color="#2B5B84", s=55, zorder=3)
    ax.scatter(null_t10, [0] * 4, color="#888888", s=55, marker="s", zorder=3)
    ax.axvline(np.mean(anch_t10), color="#2B5B84", ls="--", lw=1.4)
    ax.axvline(np.mean(null_t10), color="#888888", ls=":", lw=1.4)
    ax.text(np.mean(anch_t10) + 0.002, 1.18, f"Mean {np.mean(anch_t10):.3f}", color="#2B5B84", fontsize=8.5, fontweight="bold")
    ax.text(np.mean(null_t10) - 0.002, -0.28, f"Mean {np.mean(null_t10):.3f}", color="#555555", fontsize=8.5, ha="right", fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Null controls", "PITL anchors"], fontsize=9)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Mean cosine similarity to top-10 nearest provisions (top 3 PCs removed)", fontsize=8.5)
    ax.set_title("LegalBERT Diagnostic Check (Corpus v2.2, 1,085 provisions)", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_1_legalbert_diagnostic.png", dpi=300)
    plt.close()

    # 5. Figure 4.2: Theme Heatmap
    f42 = pd.read_csv(RESULTS_DIR / "figure_4_2_thematic_shares.csv")
    j_cols = ["AU†", "CN", "EU", "ID", "UK", "US-F†", "US-S†"]
    j_labels = ["Australia†", "China", "European Union", "Indonesia", "United Kingdom", "US (federal)†", "US (state)†"]
    mat = np.zeros((len(f42), 7))
    for i, r in f42.iterrows():
        for j, c in enumerate(j_cols):
            mat[i, j] = float(str(r[c]).split("%")[0])
    y_labels = [f"{r['theme']} (n={r['n_provisions']})" for _, r in f42.iterrows()]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=300)
    ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(7))
    ax.set_xticklabels(j_labels, fontsize=8.5, rotation=20, ha="right")
    ax.set_yticks(range(len(f42)))
    ax.set_yticklabels(y_labels, fontsize=8.3)
    for i in range(len(f42)):
        for j in range(7):
            v = mat[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.3,
                        color="white" if v > 50 else "#1A2B3C", fontweight="bold" if v >= 30 else "normal")
    ax.set_title("Jurisdictional Theme Shares (%) — NMF D-1b (k = 8, ARI = 0.922, span 62.5%)", fontsize=9.2, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_2_theme_heatmap.png", dpi=300)
    plt.close()


    # 7. Figure 4.4: Indication Matrix
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    j_mat_labels = ["Indonesia (full reading)", "China (CN-05 full reading)", "European Union", "United Kingdom", "Australia†", "United States†"]
    grid = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.0, 0.0]])
    cell_txt = [
        ["Partial (PP 17/2025 Art. 13)", "Partial (PP 17/2025 Art. 21(2))"],
        ["Partial (CN-05 Arts. 13–14;\nCN-04 Arts. 19, 40)", "Partial (CN-04 Art. 31)"],
        ["None identified", "Partial (GDPR Art. 8, consent)"],
        ["General (OSA s. 75)", "Partial (OSA s. 75, post-mortem)"],
        ["None identified", "General (SMMA Sch. 1)"],
        ["None identified", "None identified"],
    ]
    cmap = plt.matplotlib.colors.ListedColormap(["#F2F2F2", "#FCE8B2", "#B7DFCB"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Component C1\n(Bounded harm alert)", "Component C2\n(Verified supervision)"], fontsize=9, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(j_mat_labels, fontsize=8.5)
    for i in range(6):
        for j in range(2):
            ax.text(j, i, cell_txt[i][j], ha="center", va="center", fontsize=8, color="#1A2B3C",
                    fontweight="bold" if grid[i, j] == 1.0 else "normal")
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks(np.arange(0.5, 5.5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Closest Provision Identified per Component (Corpus v2.2, 21 instruments)", fontsize=9.5, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_4_pitl_indication_matrix.png", dpi=300)
    plt.close()


    plt.close()

    print("STEP 7 COMPLETE: ALL FIGURES GENERATED")


if __name__ == "__main__":
    main()
