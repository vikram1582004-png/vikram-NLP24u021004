from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from fairness import (
    exposure_by_group,
    exposure_kl_to_catalog,
    exposure_parity_gap,
    fairness_score,
)
from llm_engine import explain_recommendation, parse_preferences
from recommender import recommend

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

st.set_page_config(
    page_title="FairRank AI | Fairness-Aware LLM Recommendation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #0b1220;
        --navy-2: #121b2e;
        --blue: #2563eb;
        --blue-2: #60a5fa;
        --cyan: #06b6d4;
        --green: #10b981;
        --amber: #f59e0b;
        --red: #ef4444;
        --ink: #172033;
        --muted: #64748b;
        --line: #e7ebf2;
        --surface: #ffffff;
        --surface-2: #f7f9fc;
    }

    .stApp {
        background:
            radial-gradient(circle at 95% 5%, rgba(37,99,235,.10), transparent 24%),
            radial-gradient(circle at 5% 16%, rgba(6,182,212,.07), transparent 20%),
            #f4f7fb;
    }

    [data-testid="stHeader"] { background: rgba(244,247,251,.85); }
    .block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2rem; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #10192c 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * { color: #edf3ff !important; }
    section[data-testid="stSidebar"] .stCaptionContainer p { color: #a8b7cf !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }

    /* Sidebar controls */
    section[data-testid="stSidebar"] div[role="slider"] { accent-color: #60a5fa; }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div { background: #2563eb; }

    .brand {
        display:flex; align-items:center; gap:.8rem; margin-bottom:.5rem;
    }
    .brand-icon {
        width:42px; height:42px; border-radius:13px;
        display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg,#2563eb,#06b6d4);
        box-shadow: 0 10px 30px rgba(37,99,235,.28);
        font-size:1.35rem;
    }
    .brand-title { font-weight:800; font-size:1.08rem; letter-spacing:-.02em; }
    .brand-sub { font-size:.72rem; color:#9fb0ca; margin-top:2px; }

    .hero {
        position:relative; overflow:hidden;
        border-radius:26px; padding:2rem 2.2rem;
        background: linear-gradient(130deg, #0b1220 0%, #111c32 48%, #173a68 100%);
        box-shadow: 0 24px 70px rgba(11,18,32,.18);
        margin-bottom:1.15rem;
    }
    .hero::after {
        content:""; position:absolute; width:360px; height:360px; border-radius:50%;
        right:-120px; top:-150px; background:radial-gradient(circle, rgba(96,165,250,.30), transparent 68%);
    }
    .hero-kicker { display:inline-flex; gap:.45rem; align-items:center; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; color:#a8c8ff; font-weight:700; }
    .hero h1 { color:white; font-size:2.65rem; line-height:1.08; margin:.45rem 0 .75rem; letter-spacing:-.04em; }
    .hero p { color:#c4d2e8; font-size:1rem; max-width:900px; margin:0; }
    .hero-chip { display:inline-flex; align-items:center; gap:.45rem; margin-top:1.15rem; padding:.48rem .75rem; border-radius:999px; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.12); color:#e6eefc; font-size:.78rem; }

    .section-label { font-size:.74rem; font-weight:800; text-transform:uppercase; letter-spacing:.10em; color:#718096; margin:1.05rem 0 .55rem; }

    .query-card {
        background:rgba(255,255,255,.96); border:1px solid var(--line);
        border-radius:18px; padding:1rem 1.1rem .8rem;
        box-shadow:0 10px 30px rgba(15,23,42,.06);
    }
    .query-title { font-weight:800; color:var(--ink); margin-bottom:.2rem; }
    .query-helper { color:var(--muted); font-size:.82rem; }

    .pill {
        display:inline-flex; align-items:center; gap:.35rem; border-radius:999px;
        padding:.28rem .58rem; font-size:.72rem; font-weight:700;
        background:#eef4ff; color:#1d4ed8; border:1px solid #dce8ff;
    }
    .pill.green { background:#ecfdf5; color:#047857; border-color:#c8f3de; }
    .pill.slate { background:#f1f5f9; color:#475569; border-color:#e2e8f0; }
    .pill.amber { background:#fffbeb; color:#a16207; border-color:#fde68a; }

    /* Buttons */
    div.stButton > button {
        border-radius:12px; min-height:44px; font-weight:800;
        border:1px solid #dbe5f0; transition:all .18s ease;
        box-shadow:0 5px 14px rgba(15,23,42,.05);
    }
    div.stButton > button:hover { transform:translateY(-1px); box-shadow:0 9px 22px rgba(15,23,42,.09); }
    div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#2563eb,#1d4ed8); color:white; border:0; }

    .metric-card {
        background:linear-gradient(180deg,#ffffff 0%,#fbfcfe 100%);
        border:1px solid var(--line); border-radius:16px; padding:1rem;
        box-shadow:0 9px 26px rgba(15,23,42,.05); min-height:112px;
    }
    .metric-label { color:#64748b; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
    .metric-value { color:#111827; font-size:1.65rem; font-weight:850; margin-top:.22rem; }
    .metric-delta { font-size:.72rem; color:#64748b; margin-top:.15rem; }
    .metric-delta.good { color:#047857; }
    .metric-delta.warn { color:#b45309; }

    .rec-card {
        background:#ffffff; border:1px solid var(--line); border-radius:18px;
        padding:1rem 1rem .9rem; margin-bottom:.75rem;
        box-shadow:0 8px 24px rgba(15,23,42,.05);
    }
    .rec-card:hover { border-color:#c9d8ee; box-shadow:0 12px 30px rgba(15,23,42,.08); }
    .rank { width:34px; height:34px; border-radius:11px; background:#eff6ff; color:#2563eb; display:flex; align-items:center; justify-content:center; font-weight:850; }
    .rec-title { color:#101828; font-size:1rem; font-weight:820; }
    .company { color:#64748b; font-size:.80rem; margin-top:.12rem; }
    .mini-row { display:flex; flex-wrap:wrap; gap:.35rem; margin:.65rem 0 .55rem; }
    .score-row { display:flex; justify-content:space-between; align-items:center; font-size:.74rem; color:#64748b; margin-top:.28rem; }
    .bar { height:6px; border-radius:999px; background:#eef2f7; overflow:hidden; margin-top:.26rem; }
    .bar > div { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#06b6d4); }
    .explain { color:#475569; font-size:.79rem; line-height:1.45; margin-top:.65rem; }

    .rank-heading {
        display:flex; justify-content:space-between; align-items:center; margin:.15rem 0 .65rem;
    }
    .rank-heading h3 { margin:0; color:#172033; font-size:1.18rem; letter-spacing:-.02em; }
    .rank-heading span { color:#64748b; font-size:.75rem; }

    .compare-note {
        background:linear-gradient(135deg,#eff6ff,#f0fdfa); border:1px solid #d7e7ff;
        border-radius:16px; padding:.85rem 1rem; color:#334155; font-size:.82rem;
    }
    .pipeline {
        display:flex; gap:.55rem; align-items:center; flex-wrap:wrap; margin:1rem 0;
    }
    .step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.72rem .82rem; box-shadow:0 6px 18px rgba(15,23,42,.04); }
    .step b { font-size:.78rem; color:#172033; }
    .step small { display:block; color:#64748b; margin-top:.13rem; }
    .arrow { color:#94a3b8; font-weight:900; }

    .footer {
        margin-top:1.4rem; padding:1rem 0 .2rem; color:#7b8798; font-size:.72rem;
        border-top:1px solid #e7ebf2;
    }

    /* Keep Streamlit's default padding friendly */
    div[data-testid="stTabs"] { margin-top:.35rem; }
    button[data-baseweb="tab"] { font-weight:750; }
    textarea { border-radius:14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(BASE / "data" / "internships.csv")


catalog = load_catalog()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">⚖️ AI + RECOMMENDATION + RESPONSIBLE RANKING</div>
        <h1>FairRank AI</h1>
        <p>Fairness-aware LLM recommendation for internships — combining natural-language understanding, transparent relevance scoring, and exposure-aware re-ranking.</p>
        <div class="hero-chip">✦ Academic MVP · Synthetic data · Deterministic fairness layer</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">⚖️</div>
          <div>
            <div class="brand-title">FairRank AI</div>
            <div class="brand-sub">Minor Project · AI Recommendation</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Recommendation Controls")
    k = st.slider("Top-K recommendations", 5, 15, 10)
    fairness_strength = st.slider("Fairness strength", 0.0, 1.0, 0.30, 0.05)
    st.markdown("---")
    st.markdown("**Fairness criterion**")
    st.caption(
        "This MVP balances relevance with ranking exposure across internship-provider size groups: Startup, Small, Mid-size, and Enterprise."
    )
    st.markdown("---")
    st.markdown("**Quick demo queries**")
    q1 = st.button("🐍 Python + AI + Remote", use_container_width=True)
    q2 = st.button("☁️ Cloud + Backend", use_container_width=True)
    q3 = st.button("📊 Data + Analytics", use_container_width=True)

sample = "I know Python and C++. I am interested in AI and software development, and I prefer remote internships."
if q1:
    sample = "I know Python and C++. I am interested in AI and software development, and I prefer remote internships."
if q2:
    sample = "I want a backend internship using Python, APIs, databases and cloud technologies."
if q3:
    sample = "I am interested in data analysis, machine learning and analytics, preferably in a remote role."

# -----------------------------------------------------------------------------
# Query area
# -----------------------------------------------------------------------------
st.markdown('<div class="section-label">01 · Describe your goal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="query-card"><div class="query-title">Tell the recommender what you are looking for</div><div class="query-helper">The LLM/local parser extracts skills, domains, location, work mode and eligibility signals.</div></div>',
    unsafe_allow_html=True,
)
user_text = st.text_area("", value=sample, height=110, label_visibility="collapsed")

c1, c2, c3 = st.columns([1.15, 1.25, 1.6])
with c1:
    run = st.button("✨ Generate recommendations", type="primary", use_container_width=True)
with c2:
    clear = st.button("↻ Reset demo", use_container_width=True)
with c3:
    st.markdown('<div style="padding:.55rem 0;color:#64748b;font-size:.82rem;">LLM intent extraction stays separate from the transparent ranking + fairness layer.</div>', unsafe_allow_html=True)

if clear:
    st.session_state.pop("last_result", None)
    st.rerun()

if run or "last_result" in st.session_state:
    if run:
        prefs, mode_label = parse_preferences(user_text)
        baseline, fair = recommend(catalog, prefs, k=k, fairness_strength=fairness_strength)
        st.session_state.last_result = {
            "prefs": prefs,
            "mode": mode_label,
            "baseline": baseline,
            "fair": fair,
            "user_text": user_text,
        }

    result = st.session_state.last_result
    prefs = result["prefs"]
    baseline = result["baseline"]
    fair = result["fair"]

    mode_class = "green" if "LLM" in result["mode"] else "amber"
    st.markdown(
        f'<div style="margin:.9rem 0 .55rem;"><span class="pill {mode_class}">● {result["mode"]}</span> <span class="pill slate">{len(catalog)} opportunities in catalog</span></div>',
        unsafe_allow_html=True,
    )

    # Preference chips
    chips = []
    chips += [f"Skill: {x}" for x in prefs.skills[:5]]
    chips += [f"Domain: {x}" for x in prefs.domains[:4]]
    chips += [f"Mode: {x}" for x in prefs.work_mode[:2]]
    chips += [f"Location: {x}" for x in prefs.locations[:2]]
    if prefs.min_cgpa is not None:
        chips.append(f"Min CGPA: {prefs.min_cgpa:g}")
    chip_html = " ".join([f'<span class="pill">{c}</span>' for c in chips]) or '<span class="pill slate">No explicit constraints detected</span>'
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:.38rem;margin:.35rem 0 .8rem;">{chip_html}</div>', unsafe_allow_html=True)

    # KPI strip
    kpi = st.columns(5)
    values = [
        ("Skills", str(len(prefs.skills)), "Detected signals"),
        ("Domains", str(len(prefs.domains)), "Interest areas"),
        ("Locations", str(len(prefs.locations)), "Location signals"),
        ("Work mode", str(len(prefs.work_mode)), "Mode signals"),
        ("Min CGPA", "Any" if prefs.min_cgpa is None else f"≤ {prefs.min_cgpa:g}", "Eligibility"),
    ]
    for col, (label, value, sub) in zip(kpi, values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "⚖️ Fairness Comparison", "🧩 How It Works"])

    with tab1:
        st.markdown('<div class="section-label">02 · Compare ranking strategies</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        def render_recommendation_cards(df: pd.DataFrame, fair_mode: bool) -> None:
            for _, row in df.iterrows():
                relevance_pct = max(0, min(100, float(row["relevance"]) * 100))
                group_class = "green" if fair_mode else "slate"
                group = str(row["company_size"])
                explanation = explain_recommendation(row.to_dict(), result["user_text"], result["mode"])
                st.markdown(
                    f"""
                    <div class="rec-card">
                      <div style="display:flex;gap:.7rem;align-items:flex-start;">
                        <div class="rank">{int(row['rank'])}</div>
                        <div style="flex:1;">
                          <div class="rec-title">{row['title']}</div>
                          <div class="company">{row['company']}</div>
                        </div>
                        <span class="pill {group_class}">{group}</span>
                      </div>
                      <div class="mini-row">
                        <span class="pill slate">{row['domain']}</span>
                        <span class="pill slate">{row['mode']}</span>
                        <span class="pill slate">{row['location']}</span>
                      </div>
                      <div class="score-row"><span>Relevance</span><b>{row['relevance']:.3f}</b></div>
                      <div class="bar"><div style="width:{relevance_pct:.1f}%"></div></div>
                      <div class="score-row"><span>Popularity</span><b>{row['popularity']:.0f}</b></div>
                      <div class="explain">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with c1:
            st.markdown('<div class="rank-heading"><h3>Baseline ranking</h3><span>Relevance-focused</span></div>', unsafe_allow_html=True)
            render_recommendation_cards(baseline, False)
        with c2:
            st.markdown('<div class="rank-heading"><h3>Fairness-aware ranking</h3><span>Relevance + exposure</span></div>', unsafe_allow_html=True)
            render_recommendation_cards(fair, True)

    with tab2:
        st.markdown('<div class="section-label">03 · Measure fairness and utility</div>', unsafe_allow_html=True)
        baseline_gap = exposure_parity_gap(baseline)
        fair_gap = exposure_parity_gap(fair)
        baseline_kl = exposure_kl_to_catalog(baseline, catalog)
        fair_kl = exposure_kl_to_catalog(fair, catalog)
        baseline_fs = fairness_score(baseline, catalog)
        fair_fs = fairness_score(fair, catalog)
        avg_base_rel = float(baseline["relevance"].mean()) if not baseline.empty else 0.0
        avg_fair_rel = float(fair["relevance"].mean()) if not fair.empty else 0.0

        m = st.columns(4)
        metric_data = [
            ("Exposure Gap", fair_gap, fair_gap - baseline_gap, "Lower is better", fair_gap <= baseline_gap),
            ("Exposure KL", fair_kl, fair_kl - baseline_kl, "Lower is better", fair_kl <= baseline_kl),
            ("Fairness Score", fair_fs, fair_fs - baseline_fs, "Higher is better", fair_fs >= baseline_fs),
            ("Mean Relevance", avg_fair_rel, avg_fair_rel - avg_base_rel, "Utility retained", avg_fair_rel >= avg_base_rel),
        ]
        for col, (label, value, delta, helper, good) in zip(m, metric_data):
            with col:
                delta_text = f"{delta:+.3f} vs baseline"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value:.3f}</div><div class="metric-delta {"good" if good else "warn"}">{delta_text}</div><div class="metric-delta">{helper}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="compare-note"><b>How to read this:</b> the fairness-aware ranker aims to reduce concentration of ranking exposure across provider groups while preserving recommendation relevance. The fairness objective is specific to this academic MVP.</div>',
            unsafe_allow_html=True,
        )

        base_exp = exposure_by_group(baseline).rename("Exposure").reset_index()
        base_exp.columns = ["company_size", "Exposure"]
        base_exp["Ranking"] = "Baseline"
        fair_exp = exposure_by_group(fair).rename("Exposure").reset_index()
        fair_exp.columns = ["company_size", "Exposure"]
        fair_exp["Ranking"] = "Fairness-aware"
        comp = pd.concat([base_exp, fair_exp], ignore_index=True)

        fig = px.bar(
            comp,
            x="company_size",
            y="Exposure",
            color="Ranking",
            barmode="group",
            text_auto=".2f",
            title="Ranking exposure by internship-provider group",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="",
            font=dict(color="#334155"),
            xaxis_title="Provider group",
            yaxis_title="Exposure share",
        )
        fig.update_yaxes(gridcolor="#e8edf5", zeroline=False)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-label">04 · Transparent system pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pipeline">
              <div class="step"><b>01 · User query</b><small>Natural language</small></div><div class="arrow">→</div>
              <div class="step"><b>02 · LLM / parser</b><small>Preference extraction</small></div><div class="arrow">→</div>
              <div class="step"><b>03 · Relevance</b><small>TF-IDF + signals</small></div><div class="arrow">→</div>
              <div class="step"><b>04 · Baseline</b><small>Initial ranking</small></div><div class="arrow">→</div>
              <div class="step"><b>05 · Fair re-rank</b><small>Exposure-aware</small></div><div class="arrow">→</div>
              <div class="step"><b>06 · Metrics</b><small>Fairness + utility</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### Why this architecture?")
            st.write("The project separates language understanding from the final ranking decision. That makes the fairness intervention explicit, testable, and reproducible.")
            st.markdown("### Fairness definition")
            st.write("The current MVP measures provider-group exposure in the Top-K list. Exposure Parity Gap is the difference between the most- and least-exposed groups.")
        with right:
            st.markdown("### What the LLM does")
            st.write("The LLM/local parser extracts user intent such as skills, domains, location, work mode, and CGPA constraints.")
            st.markdown("### What the LLM does not do")
            st.write("It does not decide the fairness metric or silently change the ranking policy. Those steps are implemented deterministically in Python.")

st.markdown(
    '<div class="footer"><b>FairRank AI</b> · Academic prototype using synthetic internship data. Not intended for real hiring/admissions decisions without domain-specific validation, governance, and human oversight.</div>',
    unsafe_allow_html=True,
)
