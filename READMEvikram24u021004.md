# FairRank AI — Fairness-Aware LLM Recommendation Platform

A responsible, transparent internship recommendation system that balances candidate relevance with provider-group exposure. FairRank AI combines natural language preference extraction (LLM with local heuristic fallback), multi-signal relevance scoring (skills, domains, eligibility), and a deterministic, exposure-aware greedy re-ranking algorithm to mitigate ranking disparities across enterprise, mid-size, small, and startup providers.

---

## 🏛️ System Architecture

- **Interactive Dashboard**: Streamlit (wide layout, responsive metric cards, custom design system)
- **Visualization**: Plotly Express (interactive exposure distribution charts)
- **Natural Language & Preference Parser**: OpenAI API (`gpt-5.6-luna` / GPT-4o) with offline regex-based fallback (`local_parse`)
- **Relevance & Scoring Engine**: TF-IDF vectorization, cosine similarity, eligibility filtering (CGPA, location, work mode)
- **Fairness & Re-Ranking Engine**: Position-discounted exposure modeling ($1 / \log_2(\text{rank} + 1)$), Exposure Parity Gap, KL divergence to catalog, and deterministic greedy re-ranking
- **Data & Benchmarking**: Pandas, NumPy, synthetic multi-tier internship catalog & evaluation test profiles

---

## 📋 Prerequisites

Ensure you have the following installed on your machine:
- **Python** (v3.10 or newer)
- **pip** (Python package installer)
- *(Optional)* An **OpenAI API Key** if you wish to use cloud LLM preference extraction (the app works 100% offline out-of-the-box via local parsing)

---

## ⚙️ Environment Configuration

1. Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

2. Configure your environment variables in `.env`:

```env
# Optional: enable the live LLM preference-extraction/explanation layer.
# If omitted or left blank, the system automatically falls back to the local parser.
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.6-luna
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/vikram1582004-png/vikram-NLP24u021004.git
cd vikram-NLP24u021004

# Create and activate a virtual environment
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Ensure dependencies such as `streamlit`, `pandas`, `numpy`, `plotly`, `python-dotenv`, `scikit-learn`, and `openai` are installed).*

---

### 3. Launch the Streamlit Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```

- **Web Dashboard**: [http://localhost:8501](http://localhost:8501)
- **Network Access**: Displayed in your terminal upon launch

---

## 📁 Project Structure

```text
fairrank-ai/
├── app.py                      # Main Streamlit web application & user interface
├── fairness.py                 # Exposure metrics, position discounts & fair re-ranking logic
├── recommender.py              # Candidate relevance matching, filtering & baseline ranking
├── llm_engine.py               # LLM preference parser, explanations & fallback heuristic parser
├── evaluate.py                 # Automated batch benchmark script for evaluation profiles
│
├── data/
│   ├── internships.csv         # Synthetic internship opportunities catalog
│   └── test_profiles.csv       # Test user profile queries for automated evaluation
│
├── outputs/
│   └── evaluation_results.csv  # Benchmark output containing comparative metrics
│
├── .env.example                # Template for environment variables
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Useful Scripts & Testing

### 1. Run the Evaluation Benchmark

Run batch evaluation over predefined student profiles to compare baseline relevance ranking vs. fairness-aware ranking:

```bash
python evaluate.py
```

This will calculate and print summary metrics to the console and save a detailed comparison table to `outputs/evaluation_results.csv`:
- **Exposure Parity Gap**: Difference in normalized exposure between the most and least represented provider groups.
- **Exposure KL Divergence**: Distribution divergence from the underlying catalog representation.
- **Fairness Score**: Composite dashboard index ($1 / (1 + 2 \cdot \text{Gap} + \text{KL})$).
- **Mean Relevance**: Demonstrates recommendation utility retention before and after fair re-ranking.

---

## 🧩 How It Works

The platform enforces a strict separation of concerns between natural language parsing and decision ranking:

```text
[User Prompt]
      │
      ▼
[LLM / Local Parser] ──► Extracts skills, domains, location, mode & min CGPA
      │
      ▼
[Relevance Engine]   ──► TF-IDF & constraint filtering produce candidate pool
      │
      ▼
[Baseline Ranker]    ──► Top-K ranked purely by relevance
      │
      ▼
[Fair Re-Ranker]     ──► Greedy exposure balancing: Score = Relevance + λ * Deficit
      │
      ▼
[Interactive UI]     ──► Side-by-side cards, trade-off metrics & exposure charts
```

1. **User Intent Extraction**: Extracts structured signals (skills, domains, location preference, work mode, eligibility) without letting the LLM tamper with ranking rules.
2. **Relevance Calculation**: Transparent matching against the internship database using TF-IDF and structured constraint validation.
3. **Exposure Balancing**: Applies logarithmic position discounting ($1 / \log_2(\text{rank} + 1)$) and boosts under-represented provider tiers (Startup, Small, Mid-size, Enterprise) based on the user-controlled fairness strength slider ($\lambda_{\text{fair}}$).

---

## 🛡️ Responsible AI & Ethical Disclaimer

**FairRank AI** is an academic research prototype developed with synthetic internship data. The fairness objective here optimizes provider-side ranking exposure across company tiers; it is a proof-of-concept design choice rather than an all-encompassing definition of fairness. It is not intended for high-stakes hiring or admissions decisions without domain-specific validation, compliance auditing, and human oversight.
