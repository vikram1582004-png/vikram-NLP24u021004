from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairness import exposure_parity_gap, exposure_kl_to_catalog, fairness_score
from llm_engine import local_parse
from recommender import recommend

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "internships.csv"
PROFILES = BASE / "data" / "test_profiles.csv"
OUT = BASE / "outputs" / "evaluation_results.csv"


def main() -> None:
    catalog = pd.read_csv(DATA)
    profiles = pd.read_csv(PROFILES)
    rows = []
    for _, p in profiles.iterrows():
        prefs = local_parse(str(p["profile"]))
        baseline, fair = recommend(catalog, prefs, k=10, fairness_strength=0.30)
        rows.append(
            {
                "profile_id": p["profile_id"],
                "baseline_exposure_gap": exposure_parity_gap(baseline),
                "fair_exposure_gap": exposure_parity_gap(fair),
                "baseline_exposure_kl": exposure_kl_to_catalog(baseline, catalog),
                "fair_exposure_kl": exposure_kl_to_catalog(fair, catalog),
                "baseline_fairness_score": fairness_score(baseline, catalog),
                "fair_fairness_score": fairness_score(fair, catalog),
                "baseline_relevance": baseline["relevance"].mean(),
                "fair_relevance": fair["relevance"].mean(),
            }
        )

    results = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)

    summary = {
        "profiles": len(results),
        "mean_exposure_gap_baseline": results["baseline_exposure_gap"].mean(),
        "mean_exposure_gap_fair": results["fair_exposure_gap"].mean(),
        "mean_exposure_kl_baseline": results["baseline_exposure_kl"].mean(),
        "mean_exposure_kl_fair": results["fair_exposure_kl"].mean(),
        "mean_fairness_score_baseline": results["baseline_fairness_score"].mean(),
        "mean_fairness_score_fair": results["fair_fairness_score"].mean(),
        "mean_relevance_baseline": results["baseline_relevance"].mean(),
        "mean_relevance_fair": results["fair_relevance"].mean(),
    }
    print(results.round(4).to_string(index=False))
    print("\nSummary")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
