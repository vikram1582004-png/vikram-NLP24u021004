"""Optional LLM layer.

When OPENAI_API_KEY is configured, the app uses an OpenAI model to turn the
natural-language request into structured preferences and to generate concise
recommendation explanations. Without a key, a deterministic local parser keeps
the prototype fully runnable offline.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from recommender import ParsedPreferences

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


KNOWN_SKILLS = [
    "python", "c++", "java", "javascript", "typescript", "sql", "react", "node.js",
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "pandas", "numpy", "tensorflow", "pytorch", "git", "docker", "aws", "mongodb",
]
KNOWN_DOMAINS = [
    "ai", "artificial intelligence", "software", "web development", "data science",
    "cybersecurity", "cloud", "mobile", "backend", "frontend", "nlp", "computer vision",
]
KNOWN_LOCATIONS = ["remote", "bengaluru", "bangalore", "delhi", "mumbai", "hyderabad", "pune", "indore"]
KNOWN_MODES = ["remote", "hybrid", "onsite", "on-site"]
KNOWN_COMPANY_SIZES = ["startup", "small", "mid-size", "enterprise"]


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", text.lower()) is not None


def local_parse(user_text: str) -> ParsedPreferences:
    text = user_text.lower()
    skills = [s for s in KNOWN_SKILLS if _contains(text, s)]
    domains = [d for d in KNOWN_DOMAINS if _contains(text, d)]
    locations = [l for l in KNOWN_LOCATIONS if _contains(text, l)]
    modes = [m for m in KNOWN_MODES if _contains(text, m)]
    sizes = [s for s in KNOWN_COMPANY_SIZES if _contains(text, s)]

    cgpa = None
    match = re.search(r"(?:cgpa|gpa)[^0-9]{0,8}([0-9](?:\.[0-9]+)?)", text)
    if match:
        try:
            cgpa = float(match.group(1))
        except ValueError:
            cgpa = None

    # Domain inference from common natural-language cues.
    if "ai" in text or "artificial intelligence" in text or "machine learning" in text:
        domains += ["ai"] if "ai" not in domains else []
    if "software" in text or "developer" in text or "development" in text:
        domains += ["software"] if "software" not in domains else []
    if "remote" in text and "remote" not in modes:
        modes.append("remote")

    return ParsedPreferences(
        skills=sorted(set(skills)),
        domains=sorted(set(domains)),
        locations=sorted(set(locations)),
        work_mode=sorted(set(modes)),
        preferred_company_sizes=sorted(set(sizes)),
        min_cgpa=cgpa,
    )


def parse_preferences(user_text: str) -> tuple[ParsedPreferences, str]:
    """Return parsed preferences and the mode used (LLM or local)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return local_parse(user_text), "Local fallback"

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)
    prompt = f"""
Extract recommendation preferences from this internship request.
Return ONLY valid JSON with keys: skills, domains, locations, work_mode,
preferred_company_sizes, min_cgpa. Each list must contain concise strings.
Use an empty list when not present and null for min_cgpa.
User request: {user_text}
""".strip()

    try:
        response = client.responses.create(model=model, input=prompt)
        raw = response.output_text.strip()
        payload: dict[str, Any] = json.loads(raw)
        prefs = ParsedPreferences(
            skills=list(payload.get("skills") or []),
            domains=list(payload.get("domains") or []),
            locations=list(payload.get("locations") or []),
            work_mode=list(payload.get("work_mode") or []),
            preferred_company_sizes=list(payload.get("preferred_company_sizes") or []),
            min_cgpa=float(payload["min_cgpa"]) if payload.get("min_cgpa") is not None else None,
        )
        return prefs, f"LLM ({model})"
    except Exception:
        return local_parse(user_text), "Local fallback after LLM error"


def explain_recommendation(row: dict[str, Any], user_text: str, mode_label: str) -> str:
    """Generate an explanation; LLM mode is optional, local explanation is deterministic."""
    if mode_label.startswith("LLM") and os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = (
            "Explain in one sentence why this internship is relevant to the user. "
            "Do not mention fairness or protected attributes. Be factual and concise.\n"
            f"User: {user_text}\n"
            f"Internship: {row}"
        )
        try:
            response = client.responses.create(model=model, input=prompt)
            return response.output_text.strip()
        except Exception:
            pass

    reasons = []
    skills = str(row.get("skills", ""))
    if skills:
        reasons.append(f"matches skills such as {skills.split(';')[0].strip()}")
    if row.get("mode"):
        reasons.append(f"is {str(row['mode']).lower()}")
    if row.get("domain"):
        reasons.append(f"fits the {str(row['domain']).lower()} domain")
    return "Recommended because it " + ", ".join(reasons[:2]) + "."
