#!/usr/bin/env python3
"""
CPU-only candidate ranker for the Redrob Data & AI challenge.

The scorer is intentionally transparent: it encodes the job description as a
rubric, reads career evidence more heavily than skill tags, applies behavioral
availability modifiers, and generates reasoning from the same features used for
ranking.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import heapq
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable


REFERENCE_DATE = date(2026, 6, 27)

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "tech mahindra",
    "l&t infotech",
    "ltimindtree",
    "hexaware",
    "mphasis",
    "hcltech",
    "hcl technologies",
}

PRODUCT_COMPANIES = {
    "cred",
    "swiggy",
    "zomato",
    "flipkart",
    "razorpay",
    "freshworks",
    "zoho",
    "hooli",
    "initech",
    "stark industries",
    "uber",
    "atlassian",
    "phonepe",
    "ola",
    "paytm",
    "meesho",
    "dunzo",
    "postman",
    "microsoft",
    "google",
    "amazon",
    "meta",
    "nvidia",
    "cohere",
    "anthropic",
    "openai",
}

TIER1_INDIAN_LOCATIONS = {
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
    "mumbai",
    "hyderabad",
    "bengaluru",
    "bangalore",
}

STRONG_TITLES = {
    "ai engineer",
    "senior ai engineer",
    "ml engineer",
    "senior machine learning engineer",
    "machine learning engineer",
    "data scientist",
    "nlp engineer",
    "search engineer",
    "backend engineer",
    "data engineer",
    "analytics engineer",
    "software engineer",
}

ADJACENT_TITLES = {
    "full stack developer",
    "cloud engineer",
    "platform engineer",
    "mobile developer",
    "devops engineer",
}

NON_TECH_TITLES = {
    "marketing manager",
    "hr manager",
    "accountant",
    "sales executive",
    "customer support",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "operations manager",
    "project manager",
    "business analyst",
    "content writer",
}

GROUPS = {
    "retrieval": [
        "retrieval",
        "information retrieval",
        "ranking",
        "ranker",
        "search",
        "recommendation",
        "recommender",
        "matching",
        "bm25",
        "hybrid search",
        "semantic search",
        "vector search",
        "vector database",
        "faiss",
        "pinecone",
        "qdrant",
        "weaviate",
        "milvus",
        "opensearch",
        "elasticsearch",
        "rerank",
        "reranker",
        "cross-encoder",
        "bi-encoder",
        "colbert",
        "dense retrieval",
        "sparse retrieval",
        "splade",
        "vespa",
    ],
    "llm_nlp": [
        "llm",
        "rag",
        "fine-tuning",
        "finetuning",
        "lora",
        "qlora",
        "peft",
        "transformer",
        "bert",
        "nlp",
        "language model",
        "embeddings",
        "sentence-transformers",
        "bge",
        "e5",
        "vllm",
        "triton",
        "tensorrt",
        "langgraph",
        "llamaindex",
        "haystack",
        "sft",
        "dpo",
        "rlhf",
        "gguf",
        "ollama",
    ],
    "production": [
        "production",
        "deployed",
        "real users",
        "scale",
        "latency",
        "monitoring",
        "model serving",
        "inference",
        "pipeline",
        "feature store",
        "feature pipeline",
        "a/b",
        "ab test",
        "online",
        "offline benchmark",
    ],
    "evaluation": [
        "evaluation",
        "eval",
        "ndcg",
        "mrr",
        "map",
        "precision",
        "recall",
        "benchmark",
        "experiment",
        "quality regression",
        "ragas",
        "deepeval",
        "trulens",
        "lm-eval",
    ],
    "python": [
        "python",
        "pytorch",
        "tensorflow",
        "scikit",
        "sklearn",
        "pandas",
        "numpy",
        "fastapi",
        "flask",
        "spark",
        "airflow",
        "kafka",
    ],
    "cv_speech": [
        "computer vision",
        "image classification",
        "object detection",
        "gan",
        "speech recognition",
        "tts",
        "robotics",
    ],
    "framework_demo": [
        "langchain",
        "openai api",
        "anthropic api",
        "chatgpt",
        "prompt",
        "side project",
        "tutorial",
    ],
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.82,
    "expert": 1.0,
}

SINGLE_TOKEN_TERMS = {
    group: [term for term in terms if re.fullmatch(r"[a-z0-9]+", term)]
    for group, terms in GROUPS.items()
}

PHRASE_TERMS = {
    group: [term for term in terms if not re.fullmatch(r"[a-z0-9]+", term)]
    for group, terms in GROUPS.items()
}


@dataclass
class Evidence:
    candidate_id: str
    name: str
    title: str
    location: str
    country: str
    years: float
    companies: list[str]
    current_company: str
    current_industry: str
    current_company_size: str
    skills: list[dict]
    signals: dict
    text_all: str
    career_text: str
    skill_names: set[str]
    group_hits_all: dict[str, int] = field(default_factory=dict)
    group_hits_career: dict[str, int] = field(default_factory=dict)
    ai_skill_count: int = 0
    suspicious_skill_count: int = 0
    product_months: int = 0
    services_months: int = 0
    current_role_months: int = 0
    impossible_timeline: bool = False


@dataclass
class ScoreBreakdown:
    final: float
    role: float
    production: float
    career: float
    skills: float
    logistics: float
    behavior: float
    consistency: float
    penalty: float
    labels: list[str]


def open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def to_int(val: object, default: int = 0) -> int:
    if val is None:
        return default
    try:
        s = re.sub(r"[^\d]", "", str(val))
        return int(s) if s else default
    except Exception:
        return default


def to_float(val: object, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        try:
            s = re.sub(r"[^\d\.]", "", str(val))
            return float(s) if s else default
        except Exception:
            return default


def clean(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def token_counts(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text))


def count_hits(text: str, tokens: Counter, phrases: Iterable[str]) -> int:
    total = 0
    for phrase in phrases:
        p = phrase.lower()
        if re.fullmatch(r"[a-z0-9]+", p):
            total += tokens[p]
        else:
            total += text.count(p)
    return total


def count_group_hits(text: str, tokens: Counter) -> dict[str, int]:
    hits: dict[str, int] = {}
    for group in GROUPS:
        total = sum(tokens[term] for term in SINGLE_TOKEN_TERMS[group])
        total += sum(text.count(term) for term in PHRASE_TERMS[group])
        hits[group] = total
    return hits


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        y, m, d = value.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def recency_score(value: str | None) -> float:
    d = parse_date(value)
    if not d:
        return 0.2
    days = max(0, (REFERENCE_DATE - d).days)
    if days <= 14:
        return 1.0
    if days <= 45:
        return 0.85
    if days <= 90:
        return 0.65
    if days <= 180:
        return 0.35
    return 0.12


def build_evidence(candidate: dict) -> Evidence:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    career_bits = []
    companies = []
    product_months = 0
    services_months = 0
    current_role_months = 0
    impossible_timeline = False

    for job in career:
        company = clean(job.get("company"))
        companies.append(company)
        title = clean(job.get("title"))
        desc = clean(job.get("description"))
        career_bits.append(" ".join([company, title, clean(job.get("industry")), desc]))
        months = to_int(job.get("duration_months"), 0)
        is_service_heuristic = any(
            term in company for term in ["consultancy", "technologies ltd", "solutions", "services", "infotech", "systems ltd"]
        )
        if company in PRODUCT_COMPANIES:
            product_months += months
        if company in SERVICES_COMPANIES or is_service_heuristic:
            services_months += months
        if job.get("is_current"):
            current_role_months = months
        if months == 0 and job.get("start_date") != job.get("end_date"):
            impossible_timeline = True
        if job.get("end_date") is None and not job.get("is_current"):
            impossible_timeline = True

    skill_names = {clean(s.get("name")) for s in skills}
    skill_text = " ".join(
        f"{clean(s.get('name'))} {clean(s.get('proficiency'))}" for s in skills
    )
    profile_text = " ".join(
        clean(profile.get(k))
        for k in [
            "headline",
            "summary",
            "location",
            "country",
            "current_title",
            "current_company",
            "current_industry",
        ]
    )
    career_text = " ".join(career_bits)
    text_all = " ".join([profile_text, career_text, skill_text])
    all_tokens = token_counts(text_all)
    career_tokens = token_counts(career_text)

    group_hits_all = count_group_hits(text_all, all_tokens)
    group_hits_career = count_group_hits(career_text, career_tokens)

    ai_terms = set(GROUPS["retrieval"] + GROUPS["llm_nlp"] + GROUPS["evaluation"])
    ai_skill_count = sum(1 for name in skill_names if any(term in name for term in ai_terms))
    suspicious_skill_count = sum(
        1
        for s in skills
        if clean(s.get("proficiency")) == "expert"
        and to_int(s.get("duration_months"), 0) <= 3
    )

    return Evidence(
        candidate_id=candidate.get("candidate_id", ""),
        name=profile.get("anonymized_name", ""),
        title=clean(profile.get("current_title")),
        location=clean(profile.get("location")),
        country=clean(profile.get("country")),
        years=to_float(profile.get("years_of_experience"), 0.0),
        companies=companies,
        current_company=clean(profile.get("current_company")),
        current_industry=clean(profile.get("current_industry")),
        current_company_size=clean(profile.get("current_company_size")),
        skills=skills,
        signals=signals,
        text_all=text_all,
        career_text=career_text,
        skill_names=skill_names,
        group_hits_all=group_hits_all,
        group_hits_career=group_hits_career,
        ai_skill_count=ai_skill_count,
        suspicious_skill_count=suspicious_skill_count,
        product_months=product_months,
        services_months=services_months,
        current_role_months=current_role_months,
        impossible_timeline=impossible_timeline,
    )


def score_title(e: Evidence) -> float:
    if e.title in STRONG_TITLES:
        return 1.0
    if any(t in e.title for t in STRONG_TITLES):
        return 0.9
    if e.title in ADJACENT_TITLES:
        return 0.55
    if e.title in NON_TECH_TITLES:
        return 0.05
    if any(term in e.title for term in ["engineer", "scientist", "researcher", "developer"]):
        return 0.4
    return 0.35


def score_experience(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.84
    if 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return 0.54
    if 2.0 <= years < 3.0:
        return 0.28
    return 0.2


def score_skills(e: Evidence) -> float:
    value = 0.0
    assessment_scores = e.signals.get("skill_assessment_scores") or {}
    for s in e.skills:
        name = clean(s.get("name"))
        prof = PROFICIENCY_WEIGHT.get(clean(s.get("proficiency")), 0.4)
        months = min(to_float(s.get("duration_months"), 0.0), 60.0) / 60.0
        endorsements = min(to_float(s.get("endorsements"), 0.0), 50.0) / 50.0
        assessment = 0.0
        if name:
            assessment_raw = assessment_scores.get(name) or assessment_scores.get(name.replace(" ", ""))
            if assessment_raw is not None:
                assessment = clamp(to_float(assessment_raw, 0.0) / 100.0)
        credibility = 0.45 * prof + 0.25 * months + 0.15 * endorsements + 0.15 * assessment
        if any(term in name for term in GROUPS["retrieval"]):
            value += 2.6 * credibility
        elif any(term in name for term in GROUPS["llm_nlp"]):
            value += 1.8 * credibility
        elif any(term in name for term in GROUPS["evaluation"]):
            value += 1.6 * credibility
        elif any(term in name for term in GROUPS["python"]):
            value += 0.9 * credibility
        elif any(term in name for term in GROUPS["cv_speech"]):
            value += 0.25 * credibility
    return clamp(value / 8.5)


def score_logistics(e: Evidence) -> float:
    sig = e.signals
    loc = e.location
    score = 0.0
    if e.country == "india":
        score += 0.42
    elif sig.get("willing_to_relocate"):
        score += 0.20

    if any(city in loc for city in TIER1_INDIAN_LOCATIONS):
        score += 0.35
    elif sig.get("willing_to_relocate"):
        score += 0.18

    mode = clean(sig.get("preferred_work_mode"))
    if any(m in mode for m in ["hybrid", "flexible", "open", "any"]):
        score += 0.10
    elif mode == "onsite" and any(city in loc for city in TIER1_INDIAN_LOCATIONS):
        score += 0.08
    elif "remote" in mode:
        score += 0.04

    notice = to_int(sig.get("notice_period_days"), 180)
    if notice <= 15:
        score += 0.16
    elif notice <= 30:
        score += 0.13
    elif notice <= 60:
        score += 0.08
    elif notice <= 90:
        score += 0.03

    return clamp(score)


def score_behavior(e: Evidence) -> float:
    sig = e.signals
    response = float(sig.get("recruiter_response_rate") or 0.0)
    response_time = float(sig.get("avg_response_time_hours") or 999.0)
    profile = float(sig.get("profile_completeness_score") or 0.0) / 100.0
    views = min(float(sig.get("profile_views_received_30d") or 0.0), 80.0) / 80.0
    saves = min(float(sig.get("saved_by_recruiters_30d") or 0.0), 12.0) / 12.0
    interviews = float(sig.get("interview_completion_rate") or 0.0)
    github = float(sig.get("github_activity_score") or -1.0)
    github_score = 0.0 if github < 0 else min(github, 100.0) / 100.0
    active = recency_score(sig.get("last_active_date"))
    fast_reply = 1.0 - min(response_time, 240.0) / 240.0
    verified = (
        int(bool(sig.get("verified_email")))
        + int(bool(sig.get("verified_phone")))
        + int(bool(sig.get("linkedin_connected")))
    ) / 3.0
    profile_strength = 0.55 * profile + 0.25 * verified + 0.20 * (1.0 if sig.get("profile_completeness_score", 0) >= 70 else 0.0)

    raw = (
        0.18 * active
        + 0.20 * response
        + 0.10 * fast_reply
        + 0.14 * profile_strength
        + 0.08 * views
        + 0.12 * saves
        + 0.08 * interviews
        + 0.06 * github_score
        + 0.04 * verified
    )
    if sig.get("open_to_work_flag"):
        raw += 0.08
    return clamp(raw)


def score_candidate(e: Evidence) -> ScoreBreakdown:
    retrieval_career = e.group_hits_career["retrieval"]
    retrieval_all = e.group_hits_all["retrieval"]
    llm_career = e.group_hits_career["llm_nlp"]
    llm_all = e.group_hits_all["llm_nlp"]
    prod_hits = e.group_hits_career["production"]
    eval_hits = e.group_hits_career["evaluation"]
    python_hits = e.group_hits_all["python"]

    role = clamp(
        0.28 * score_title(e)
        + 0.26 * clamp(retrieval_career / 3.0)
        + 0.14 * clamp(llm_career / 3.0)
        + 0.12 * clamp(retrieval_all / 5.0)
        + 0.10 * clamp(llm_all / 6.0)
        + 0.10 * clamp(python_hits / 5.0)
    )

    production = clamp(
        0.36 * clamp(prod_hits / 4.0)
        + 0.24 * clamp(eval_hits / 2.0)
        + 0.20 * (1.0 if e.product_months >= 18 else e.product_months / 18.0)
        + 0.10 * (1.0 if e.current_role_months >= 12 else e.current_role_months / 12.0)
        + 0.10 * (1.0 if e.group_hits_career["production"] >= 2 else 0.0)
    )

    career = clamp(
        0.38 * score_experience(e.years)
        + 0.22 * score_title(e)
        + 0.18 * (1.0 if e.product_months >= 24 else e.product_months / 24.0)
        + 0.12 * (0.0 if e.current_company in SERVICES_COMPANIES else 1.0)
        + 0.10 * (1.0 if e.current_company_size in {"201-500", "501-1000", "1001-5000"} else 0.55)
    )

    skills = score_skills(e)
    logistics = score_logistics(e)
    behavior = score_behavior(e)

    skill_claim_strength = clamp((retrieval_all + llm_all + e.ai_skill_count) / 12.0)
    career_support = clamp((retrieval_career + llm_career + prod_hits + eval_hits) / 8.0)
    consistency = clamp(
        0.42 * career_support
        + 0.24 * (1.0 - max(0.0, skill_claim_strength - career_support))
        + 0.18 * (1.0 if e.suspicious_skill_count == 0 else 0.35)
        + 0.16 * (0.35 if e.impossible_timeline else 1.0)
    )

    penalty = 0.0
    if e.title in NON_TECH_TITLES and career_support < 0.35:
        penalty += 0.22
    if e.services_months > 0 and e.product_months == 0 and e.years >= 5:
        penalty += 0.10
    if skill_claim_strength > 0.65 and career_support < 0.25:
        penalty += 0.20
    if e.group_hits_all["framework_demo"] >= 3 and career_support < 0.45:
        penalty += 0.08
    if e.group_hits_all["cv_speech"] >= 3 and retrieval_career == 0:
        penalty += 0.08
    if e.product_months == 0 and e.services_months >= 24 and e.years >= 6:
        penalty += 0.05
    if e.group_hits_career["retrieval"] == 0 and e.group_hits_career["llm_nlp"] == 0 and e.years >= 6:
        penalty += 0.05
    if e.signals.get("last_active_date") and recency_score(e.signals.get("last_active_date")) <= 0.12:
        penalty += 0.06
    if float(e.signals.get("recruiter_response_rate") or 0.0) < 0.12:
        penalty += 0.06
    if int(e.signals.get("notice_period_days") or 0) >= 120:
        penalty += 0.05
    if e.suspicious_skill_count >= 2 or e.impossible_timeline:
        penalty += 0.20

    calibrated = (
        0.30 * role
        + 0.20 * production
        + 0.17 * career
        + 0.12 * skills
        + 0.08 * logistics
        + 0.08 * behavior
        + 0.05 * consistency
    )
    if e.group_hits_career["retrieval"] >= 2 and e.group_hits_career["evaluation"] >= 1:
        calibrated += 0.02
    if e.group_hits_career["production"] >= 2 and e.product_months >= 18:
        calibrated += 0.02
    final = clamp(calibrated - penalty)

    labels = []
    if retrieval_career:
        labels.append("retrieval/search career evidence")
    if eval_hits:
        labels.append("ranking/evaluation evidence")
    if prod_hits:
        labels.append("production systems evidence")
    if e.product_months >= 18:
        labels.append("product-company experience")
    if logistics >= 0.7:
        labels.append("strong India/location fit")
    if behavior >= 0.65:
        labels.append("strong engagement signals")
    if penalty >= 0.15:
        labels.append("notable consistency/trap penalty")

    return ScoreBreakdown(
        final=final,
        role=role,
        production=production,
        career=career,
        skills=skills,
        logistics=logistics,
        behavior=behavior,
        consistency=consistency,
        penalty=penalty,
        labels=labels,
    )


def top_skill_names(e: Evidence, limit: int = 4) -> list[str]:
    relevant = []
    for s in e.skills:
        name = str(s.get("name") or "").strip()
        low = clean(name)
        if any(
            term in low
            for term in GROUPS["retrieval"] + GROUPS["llm_nlp"] + GROUPS["python"]
        ):
            relevant.append(name)
    return relevant[:limit]


def display_title(title: str) -> str:
    words = []
    acronyms = {"ai", "ml", "nlp", "llm", "rag", "devops"}
    for word in title.split():
        words.append(word.upper() if word in acronyms else word.capitalize())
    return " ".join(words)


def reasoning(e: Evidence, b: ScoreBreakdown) -> str:
    sig = e.signals
    pieces = []
    title = display_title(e.title) if e.title else "Candidate"
    loc = e.location.title() if e.location else e.country.title()
    skills = top_skill_names(e)

    if b.role >= 0.7 and b.production >= 0.45:
        evidence = ", ".join(b.labels[:2]) or "AI/search systems"
        pieces.append(
            f"{title} with {e.years:.1f} yrs and {evidence}"
        )
    elif b.role >= 0.55:
        pieces.append(
            f"{title} with {e.years:.1f} yrs and partial fit for the AI/search rubric"
        )
    else:
        pieces.append(
            f"{title} with {e.years:.1f} yrs; included for adjacent technical and availability signals"
        )

    if skills:
        pieces.append(f"relevant skills include {', '.join(skills)}")
    if loc:
        pieces.append(f"location/logistics: {loc}")

    response = float(sig.get("recruiter_response_rate") or 0.0)
    notice = int(sig.get("notice_period_days") or 0)
    active = sig.get("last_active_date", "unknown")
    pieces.append(
        f"response rate {response:.2f}, last active {active}, notice {notice} days"
    )

    concerns = []
    if b.penalty >= 0.15:
        concerns.append("profile consistency/trap signals")
    if e.current_company in SERVICES_COMPANIES and e.product_months == 0:
        concerns.append("services-heavy career")
    if notice >= 90:
        concerns.append("long notice period")
    if response < 0.2:
        concerns.append("low recruiter response")
    if e.group_hits_career["retrieval"] == 0 and b.role < 0.75:
        concerns.append("limited explicit retrieval/ranking career evidence")

    if concerns:
        pieces.append("concern: " + ", ".join(concerns[:2]))

    text = "; ".join(pieces)
    return text[:480]


def iter_candidates(path: Path) -> Iterable[dict]:
    with open_jsonl(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def rank_candidates(candidates_path: Path, limit: int) -> list[tuple[float, str, Evidence, ScoreBreakdown]]:
    heap: list[tuple[float, str, Evidence, ScoreBreakdown]] = []
    for candidate in iter_candidates(candidates_path):
        e = build_evidence(candidate)
        b = score_candidate(e)
        row = (b.final, e.candidate_id, e, b)
        if len(heap) < limit:
            heapq.heappush(heap, row)
        elif (b.final, e.candidate_id) > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, row)
    return sorted(heap, key=lambda row: (-row[0], row[1]))


def write_submission(rows: list[tuple[float, str, Evidence, ScoreBreakdown]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, _cid, e, b) in enumerate(rows, start=1):
            adjusted = max(0.0, score - (rank - 1) * 1e-7)
            writer.writerow([e.candidate_id, rank, f"{adjusted:.6f}", reasoning(e, b)])


def write_diagnostics(rows: list[tuple[float, str, Evidence, ScoreBreakdown]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "candidate_id",
                "score",
                "title",
                "location",
                "years",
                "role",
                "production",
                "career",
                "skills",
                "logistics",
                "behavior",
                "consistency",
                "penalty",
                "labels",
            ]
        )
        for rank, (score, _cid, e, b) in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    e.candidate_id,
                    f"{score:.6f}",
                    e.title,
                    e.location,
                    f"{e.years:.1f}",
                    f"{b.role:.3f}",
                    f"{b.production:.3f}",
                    f"{b.career:.3f}",
                    f"{b.skills:.3f}",
                    f"{b.logistics:.3f}",
                    f"{b.behavior:.3f}",
                    f"{b.consistency:.3f}",
                    f"{b.penalty:.3f}",
                    " | ".join(b.labels),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("ranking_diagnostics.csv"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = rank_candidates(args.candidates, args.limit)
    write_submission(rows, args.out)
    write_diagnostics(rows, args.diagnostics)

    top = rows[0]
    print(
        f"Wrote {args.out} with {len(rows)} rows. "
        f"Top candidate: {top[2].candidate_id} score={top[0]:.6f}"
    )


if __name__ == "__main__":
    main()
