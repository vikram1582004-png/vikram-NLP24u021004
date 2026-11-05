"""Fairness metrics and fairness-aware re-ranking.

This prototype treats fairness as item/provider-side exposure balance across
internship-provider size groups. The chosen criterion is a project design
choice, not a universal definition of fairness.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


GROUP_COL = "company_size"


def position_discount(rank: int) -> float:
    """Logarithmic ranking exposure discount for 1-indexed ranks."""
    return 1.0 / np.log2(rank + 1)


def exposure_by_group(ranked: pd.DataFrame, group_col: str = GROUP_COL) -> pd.Series:
    """Return normalized exposure share by group for a ranked dataframe."""
    if ranked.empty:
        return pd.Series(dtype=float)
    exposure = defaultdict(float)
    for rank, (_, row) in enumerate(ranked.reset_index(drop=True).iterrows(), start=1):
        exposure[str(row[group_col])] += position_discount(rank)
    total = sum(exposure.values())
    return pd.Series({g: v / total for g, v in sorted(exposure.items())}) if total else pd.Series(dtype=float)


def exposure_parity_gap(ranked: pd.DataFrame, group_col: str = GROUP_COL) -> float:
    """Max-min normalized exposure across groups; lower is closer to parity."""
    shares = exposure_by_group(ranked, group_col)
    if shares.empty:
        return 0.0
    return float(shares.max() - shares.min())


def catalog_group_shares(catalog: pd.DataFrame, group_col: str = GROUP_COL) -> pd.Series:
    """Share of all catalog items in each group."""
    return catalog[group_col].astype(str).value_counts(normalize=True).sort_index()


def exposure_kl_to_catalog(ranked: pd.DataFrame, catalog: pd.DataFrame, group_col: str = GROUP_COL) -> float:
    """KL divergence of top-k exposure share from catalog composition.

    This is a descriptive concentration metric, not a claim that catalog parity
    is always the correct fairness target.
    """
    exp = exposure_by_group(ranked, group_col)
    cat = catalog_group_shares(catalog, group_col)
    groups = sorted(set(exp.index).union(cat.index))
    eps = 1e-9
    p = np.array([float(exp.get(g, 0.0)) + eps for g in groups])
    q = np.array([float(cat.get(g, 0.0)) + eps for g in groups])
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def group_counts(ranked: pd.DataFrame, group_col: str = GROUP_COL) -> pd.Series:
    return ranked[group_col].astype(str).value_counts().sort_index()


def fairness_score(ranked: pd.DataFrame, catalog: pd.DataFrame, group_col: str = GROUP_COL) -> float:
    """Convenience score: 1 means zero exposure parity gap and zero KL.

    It is intentionally just a dashboard score for the prototype, not a
    scientifically universal fairness score.
    """
    gap = exposure_parity_gap(ranked, group_col)
    kl = exposure_kl_to_catalog(ranked, catalog, group_col)
    score = 1.0 / (1.0 + 2.0 * gap + kl)
    return float(score)


def fair_rerank(
    baseline_candidates: pd.DataFrame,
    k: int = 10,
    group_col: str = GROUP_COL,
    relevance_col: str = "relevance",
    lambda_fair: float = 0.30,
) -> pd.DataFrame:
    """Greedy exposure-aware re-ranking.

    At each position, pick an item from the group with the largest exposure
    deficit relative to equal group exposure, while retaining relevance as the
    main signal. A small-group guard prevents pathological empty-group choices.
    """
    if baseline_candidates.empty:
        return baseline_candidates.copy()

    pool = baseline_candidates.copy().reset_index(drop=True)
    pool["_group"] = pool[group_col].astype(str)
    groups = sorted(pool["_group"].unique())
    target = 1.0 / max(1, len(groups))

    selected_idx: list[int] = []
    exposure = {g: 0.0 for g in groups}

    for rank in range(1, min(k, len(pool)) + 1):
        discount = position_discount(rank)
        total_future = sum(position_discount(r) for r in range(1, k + 1))

        # Exposure deficit after adding this rank to each candidate's group.
        options = []
        for idx, row in pool.iterrows():
            if idx in selected_idx:
                continue
            g = row["_group"]
            projected = exposure[g] + discount
            projected_share = projected / (sum(exposure.values()) + discount)
            deficit = target - projected_share
            relevance = float(row[relevance_col])
            # Larger fairness bonus for underexposed groups; keep relevance dominant.
            score = relevance + lambda_fair * max(deficit, -0.25)
            options.append((score, relevance, idx))

        # Prefer the highest combined score; deterministic tie-break by relevance.
        _, _, best_idx = max(options, key=lambda t: (t[0], t[1], -t[2]))
        selected_idx.append(best_idx)
        best_group = pool.loc[best_idx, "_group"]
        exposure[best_group] += discount

    result = pool.loc[selected_idx].drop(columns=["_group"]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result
