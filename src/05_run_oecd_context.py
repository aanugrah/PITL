#!/usr/bin/env python3
"""
05_run_oecd_context.py — Verifies OECD AI Incidents Monitor trends (2020-2025) for Section 1.1 / Figure 1.1.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULTS_DIR = ROOT / "data", ROOT / "results"
YEARS = [str(y) for y in range(2020, 2026)]
ORDER = ["United States", "United Kingdom", "European Union (EU27)", "China", "Australia", "Indonesia"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("STEP 5: OECD AI INCIDENT TRENDS VERIFICATION (SECTION 1.1 / FIGURE 1.1)")

    eu_all = pd.read_csv(DATA_DIR / "oecd_eu27_all.csv", comment="#")
    eu_ch = pd.read_csv(DATA_DIR / "oecd_eu27_child.csv", comment="#")
    a = eu_all.groupby(eu_all["Date"].astype(str).str[:4])["Total Incidents & Hazards"].sum()
    c = eu_ch.groupby(eu_ch["Date"].astype(str).str[:4])["Total Incidents & Hazards"].sum()
    eu_df = pd.DataFrame([
        {"jurisdiction": "European Union (EU27)", "year": int(y), "total_incidents": int(a.get(y, 0)), "child_incidents": int(c.get(y, 0))}
        for y in YEARS
    ])

    df = pd.read_csv(DATA_DIR / "oecd_ai_incidents_filtered_child.csv")
    df["year"] = df["Date"].astype(str).str[:4]
    df = df[df["year"].isin(YEARS)]
    g5 = df.groupby(["Jurisdiction", "year"], as_index=False)[["Total_Incidents", "Child_Incidents"]].sum().rename(
        columns={"Jurisdiction": "jurisdiction", "Total_Incidents": "total_incidents", "Child_Incidents": "child_incidents"}
    )
    g5["year"] = g5["year"].astype(int)

    yearly = pd.concat([g5, eu_df], ignore_index=True)
    yearly[["total_incidents", "child_incidents"]] = yearly[["total_incidents", "child_incidents"]].astype(int)
    yearly["jurisdiction"] = pd.Categorical(yearly["jurisdiction"], categories=ORDER, ordered=True)
    yearly = yearly.sort_values(["jurisdiction", "year"]).reset_index(drop=True)
    yearly.to_csv(RESULTS_DIR / "oecd_yearly_by_jurisdiction.csv", index=False)

    trends = yearly[yearly["year"].isin([2022, 2025])].groupby("year", as_index=False)[["total_incidents", "child_incidents"]].sum()
    assert tuple(trends.loc[trends["year"] == 2022, ["total_incidents", "child_incidents"]].iloc[0]) == (705, 56)
    assert tuple(trends.loc[trends["year"] == 2025, ["total_incidents", "child_incidents"]].iloc[0]) == (2749, 305)
    trends.to_csv(RESULTS_DIR / "oecd_trends_2022_2025.csv", index=False)
    print("STEP 5 COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
