"""Core recommendation pipeline for the Fairness-Aware LLM Recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fairness import fair_rerank


@dataclass
class ParsedPreferences:
    skills: list[str]
    domains: list[str]
    locations: list[str]
    work_mode: list[str]
    preferred_company_sizes: list[str]
    min_cgpa: float | None = None

    @property
    def search_text(self) -> str:
        parts = self.skills + self.domains + self.locations + self.work_mode + self.preferred_company_sizes
        return " ".join(parts)


def build_document(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("title", "")),
            str(row.get("domain", "")),
            str(row.get("skills", "")),
            str(row.get("location", "")),
            str(row.get("mode", "")),
            str(row.get("description", "")),
        ]
    )


def prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    data = catalog.copy()
    data["document"] = data.apply(build_document, axis=1)
    return data


def rank_baseline(catalog: pd.DataFrame, prefs: ParsedPreferences, pool_size: int = 30) -> pd.DataFrame:
    data = prepare_catalog(catalog)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(data["document"].values)
    query = vectorizer.transform([prefs.search_text or "internship software development"])
    relevance = cosine_similarity(query, matrix).ravel()
    data["relevance"] = relevance

    # Popularity is intentionally a secondary signal in the baseline to model
    # a common recommender tendency toward already-popular items.
    pop = data["popularity"].astype(float)
    pop_norm = (pop - pop.min()) / max(pop.max() - pop.min(), 1e-9)
    data["baseline_score"] = 0.72 * data["relevance"] + 0.28 * pop_norm

    if prefs.min_cgpa is not None:
        data = data[data["min_cgpa"].astype(float) <= float(prefs.min_cgpa)]

    for field, values in [
        ("domain", prefs.domains),
        ("location", prefs.locations),
        ("mode", prefs.work_mode),
    ]:
        if values:
            patt = "|".join(map(lambda x: str(x).lower().replace("+", "\\+"), values))
            mask = data[field].astype(str).str.lower().str.contains(patt, regex=True)
            # Don't hard-filter on preferences; add relevance-style bonuses instead.
            data.loc[mask, "baseline_score"] += 0.04

    return data.sort_values(["baseline_score", "relevance"], ascending=False).head(pool_size).reset_index(drop=True)


def recommend(catalog: pd.DataFrame, prefs: ParsedPreferences, k: int = 10, fairness_strength: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_pool = rank_baseline(catalog, prefs, pool_size=max(30, k * 4))
    baseline = baseline_pool.head(k).copy()
    baseline["rank"] = np.arange(1, len(baseline) + 1)
    fair = fair_rerank(
        baseline_pool,
        k=k,
        lambda_fair=fairness_strength,
        group_col="company_size",
        relevance_col="relevance",
    )
    return baseline, fair
