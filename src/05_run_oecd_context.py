#!/usr/bin/env python3
"""
05_run_oecd_context.py

Verifies the descriptive OECD AI Incidents and Hazards Monitor trends (2020-2025)
reported in Section 1.1 and Figure 1.1 across all six analysed jurisdictions
(United States, United Kingdom, European Union [EU27], China, Australia, Indonesia).
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _load_eu27_yearly() -> pd.DataFrame:
    eu_all = pd.read_csv(DATA_DIR / "oecd_eu27_all.csv", comment="#")
    eu_ch = pd.read_csv(DATA_DIR / "oecd_eu27_child.csv", comment="#")
    eu_all["Year"] = eu_all["Date"].astype(str).str[:4]
    eu_ch["Year"] = eu_ch["Date"].astype(str).str[:4]
    a = eu_all.groupby("Year")["Total Incidents & Hazards"].sum()
    c = eu_ch.groupby("Year")["Total Incidents & Hazards"].sum()
    rows = []
    for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        rows.append({
            "jurisdiction": "European Union (EU27)",
            "year": int(yr),
            "total_incidents": int(a.get(yr, 0)),
            "child_incidents": int(c.get(yr, 0)),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP 5: OECD AI INCIDENT TRENDS VERIFICATION (SECTION 1.1 / FIGURE 1.1)")
    print("=" * 70)

    df5 = pd.read_csv(DATA_DIR / "oecd_ai_incidents_filtered_child.csv")
    df5["Year"] = df5["Date"].astype(str).str[:4]
    df5 = df5[df5["Year"].isin(["2020", "2021", "2022", "2023", "2024", "2025"])]
    g5 = (
        df5.groupby(["Jurisdiction", "Year"], as_index=False)[["Total_Incidents", "Child_Incidents"]]
        .sum()
        .rename(columns={"Jurisdiction": "jurisdiction", "Year": "year",
                         "Total_Incidents": "total_incidents", "Child_Incidents": "child_incidents"})
    )
    g5["year"] = g5["year"].astype(int)
    g5["total_incidents"] = g5["total_incidents"].astype(int)
    g5["child_incidents"] = g5["child_incidents"].astype(int)

    eu_df = _load_eu27_yearly()
    yearly = pd.concat([g5, eu_df], ignore_index=True)
    order = ["United States", "United Kingdom", "European Union (EU27)", "China", "Australia", "Indonesia"]
    yearly["jurisdiction"] = pd.Categorical(yearly["jurisdiction"], categories=order, ordered=True)
    yearly = yearly.sort_values(["jurisdiction", "year"]).reset_index(drop=True)
    yearly.to_csv(RESULTS_DIR / "oecd_yearly_by_jurisdiction.csv", index=False)

    rows = []
    for yr in [2022, 2025]:
        sub = yearly[yearly["year"] == yr]
        tot = int(sub["total_incidents"].sum())
        ch = int(sub["child_incidents"].sum())
        rows.append({"year": yr, "total_incidents": tot, "child_incidents": ch})
        print(f"  {yr} (6 jurisdictions): total={tot:,}, child_linked={ch:,}")

    assert rows[0]["total_incidents"] == 705 and rows[0]["child_incidents"] == 56
    assert rows[1]["total_incidents"] == 2749 and rows[1]["child_incidents"] == 305

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "oecd_trends_2022_2025.csv", index=False)
    print("STEP 5 COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
