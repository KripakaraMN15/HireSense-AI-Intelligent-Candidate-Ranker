"""
ui/cards.py
-----------
Presentation layer for HireSense AI candidate cards and pool banners.
All functions here are pure HTML renderers — no Gradio, no ranking logic.

Imports from rank.py:
    - reasoning       : generates candidate summary text
    - display_title   : title-cases job titles with acronym awareness
    - top_skill_names : returns top relevant skill names for a candidate
"""

from __future__ import annotations

import html as _html

from rank import reasoning, display_title, top_skill_names


# ---------------------------------------------------------------------------
# Constants used exclusively by card renderers
# ---------------------------------------------------------------------------

SIGNAL_BARS = [
    ("Role", "role", "#7c6fcd"),
    ("Production", "production", "#10b981"),
    ("Career", "career", "#f59e0b"),
    ("Skills", "skills", "#3b82f6"),
    ("Logistics", "logistics", "#8b5cf6"),
    ("Behavior", "behavior", "#ec4899"),
]

EVIDENCE_GROUPS = ["retrieval", "llm_nlp", "evaluation", "python", "production"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def esc(value) -> str:
    """HTML-escape a value for safe embedding."""
    return _html.escape(str(value or ""), quote=True)


def skill_name_set(e) -> set[str]:
    """Return the set of lowercase skill names for a candidate."""
    return {str(s.get("name") or "").lower() for s in e.skills}


# ---------------------------------------------------------------------------
# Sub-component renderers
# ---------------------------------------------------------------------------

def _signal_progress_bars(b) -> str:
    """Render scoring dimensions as sleek horizontal progress bars."""
    bars = []
    for label, attr, color in SIGNAL_BARS:
        value = getattr(b, attr, 0)
        pct = max(0, min(100, value * 100))
        bars.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        font-size:11px;color:#94a3b8;margin-bottom:4px;letter-spacing:0.3px;">
                <span style="font-weight:500;">{label}</span>
                <span style="font-weight:700;color:#e2e8f0;font-size:12px;">{pct:.0f}%</span>
            </div>
            <div style="background:#0f172a;border-radius:6px;height:8px;overflow:hidden;
                        box-shadow:inset 0 1px 3px rgba(0,0,0,0.4);">
                <div style="background:linear-gradient(90deg,{color}cc,{color});
                            width:{pct}%;height:100%;border-radius:6px;
                            transition:width 0.4s ease;
                            box-shadow:0 0 6px {color}66;"></div>
            </div>
        </div>
        """)
    return '<div style="margin:14px 0 4px;">' + ''.join(bars) + '</div>'


def _evidence_checklist(e) -> str:
    """Render signal evidence as a styled checklist tree."""
    labels = [
        ("retrieval", "Retrieval / Search",  e.group_hits_career.get("retrieval", 0) > 0),
        ("llm_nlp",   "LLM / NLP",           e.group_hits_career.get("llm_nlp", 0) > 0),
        ("evaluation","Evaluation Metrics",   e.group_hits_career.get("evaluation", 0) > 0),
        ("python",    "Python Proficiency",   e.group_hits_career.get("python", 0) > 0),
        ("production","Production Systems",   e.group_hits_career.get("production", 0) > 0),
    ]
    lines = []
    for idx, (_key, label, hit) in enumerate(labels):
        prefix = "└──" if idx == len(labels) - 1 else "├──"
        color = "#10b981" if hit else "#64748b"
        icon  = "✓" if hit else "✗"
        weight = "600" if hit else "400"
        lines.append(
            f'<div style="font-size:12px;color:{color};margin:3px 0;font-weight:{weight};">'
            f'<span style="color:#475569;margin-right:4px;">{prefix}</span>'
            f'<span style="color:{color};">{icon}</span>'
            f'<span style="margin-left:6px;">{label}</span></div>'
        )
    return (
        '<div style="margin:12px 0 8px;color:#cbd5e1;">'
        '<div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.8px;margin-bottom:8px;">📋 Signal Evidence</div>'
        + ''.join(lines)
        + '</div>'
    )


def _section_header(label: str, color: str = "#6366f1") -> str:
    """Render a compact section header with trailing rule."""
    return (
        f'<div style="display:flex;align-items:center;margin:14px 0 8px;">'
        f'<span style="font-size:10px;font-weight:800;color:{color};'
        f'text-transform:uppercase;letter-spacing:1px;">{label}</span>'
        f'<div style="flex:1;height:1px;background:linear-gradient(90deg,{color}44,transparent);'
        f'margin-left:10px;"></div></div>'
    )


# ---------------------------------------------------------------------------
# Pool-level computations (pure data — no Gradio dependency)
# ---------------------------------------------------------------------------

def compute_pool_stats(scored) -> dict | None:
    """Aggregate pool intelligence metrics from all scored candidates."""
    if not scored:
        return None
    n = len(scored)
    retrieval_count = sum(
        1 for _, e, _ in scored if e.group_hits_career.get("retrieval", 0) > 0
    )
    open_to_work = sum(
        1 for _, e, _ in scored if e.signals.get("open_to_work_flag")
    )
    notices = [int(e.signals.get("notice_period_days") or 0) for _, e, _ in scored]
    avg_notice = sum(notices) / n
    avg_score = sum(s for s, _, _ in scored) / n * 10

    gap_groups = ["retrieval", "llm_nlp", "evaluation"]
    zero_counts = {
        g: sum(1 for _, e, _ in scored if e.group_hits_career.get(g, 0) == 0)
        for g in gap_groups
    }
    top_skill_gap = max(zero_counts, key=zero_counts.get)

    return {
        "count": n,
        "retrieval_evidence": retrieval_count,
        "open_to_work": open_to_work,
        "avg_notice": avg_notice,
        "top_skill_gap": top_skill_gap,
        "avg_score": avg_score,
    }


def generate_pool_banner(scored) -> str:
    """Render a compact pool intelligence banner."""
    metrics = compute_pool_stats(scored)
    if not metrics:
        return ""
    return f"""
    <div style="background:linear-gradient(135deg,#1e293b,#0f1a2e);
                border-left:3px solid #6366f1;border-radius:0 8px 8px 0;
                padding:14px 18px;margin-bottom:20px;
                box-shadow:0 2px 12px rgba(99,102,241,0.1);">
      <div style="display:flex;flex-wrap:wrap;gap:8px 28px;color:#e2e8f0;font-size:13px;align-items:center;">
        <span><strong style="color:#ffffff;">{metrics['count']}</strong>
              <span style="color:#64748b;margin-left:4px;">candidates evaluated</span></span>
        <span style="color:#334155;">·</span>
        <span><strong style="color:#6ee7b7;">{metrics['retrieval_evidence']}</strong>
              <span style="color:#64748b;margin-left:4px;">retrieval evidence</span></span>
        <span style="color:#334155;">·</span>
        <span><strong style="color:#a5b4fc;">{metrics['open_to_work']}</strong>
              <span style="color:#64748b;margin-left:4px;">open to work</span></span>
        <span style="color:#334155;">·</span>
        <span><span style="color:#64748b;">Avg notice</span>
              <strong style="color:#fbbf24;margin-left:4px;">{metrics['avg_notice']:.0f}d</strong></span>
        <span style="color:#334155;">·</span>
        <span><span style="color:#64748b;">Top gap</span>
              <strong style="color:#fb923c;margin-left:4px;">{metrics['top_skill_gap']}</strong></span>
        <span style="color:#334155;">·</span>
        <span><span style="color:#64748b;">Avg score</span>
              <strong style="color:#34d399;margin-left:4px;">{metrics['avg_score']:.2f}/10</strong></span>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def calculate_confidence_score(e, b, score: float) -> int:
    """Calculate deterministic confidence score (0–100%) based on signal strength."""
    components = []

    if e.signals.get("verified_email"):
        components.append(15)
    if e.signals.get("verified_phone"):
        components.append(10)
    if e.years > 0 and len(e.companies) > 0 and len(e.skills) > 0:
        components.append(15)
    if b.role > 0.6 and b.production > 0.5:
        components.append(15)
    if e.group_hits_career.get("production", 0) > 0:
        components.append(15)
    if e.signals.get("last_active_date") and e.signals.get("last_active_date") != "N/A":
        components.append(10)
    if b.penalty < 0.1:
        components.append(10)

    career_evidence_count = sum(
        1 for key in ["retrieval", "llm_nlp", "evaluation", "python", "production"]
        if e.group_hits_career.get(key, 0) > 0
    )
    if career_evidence_count >= 3:
        components.append(5)

    base = min(100, sum(components))
    boost = score * 10 * 0.1
    return int(min(100, base + boost))


def get_confidence_color(confidence: int) -> str:
    if confidence >= 80:
        return "#10b981"
    elif confidence >= 60:
        return "#f59e0b"
    return "#ef4444"


def get_confidence_label(confidence: int) -> str:
    if confidence >= 80:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Strengths / Weaknesses
# ---------------------------------------------------------------------------

def get_strengths_weaknesses(e):
    """Extract JD-aligned strengths and missing capabilities."""
    strengths = []
    evidence_labels = {
        "retrieval":  "Retrieval",
        "llm_nlp":    "LLM / NLP",
        "evaluation": "Evaluation",
        "python":     "Python",
        "production": "Production ML",
    }
    for key, label in evidence_labels.items():
        if e.group_hits_career.get(key, 0) > 0:
            strengths.append(label)

    strengths.extend(top_skill_names(e, limit=3))

    profile_text = e.text_all.lower()
    required_capabilities = {
        "Python":     ["python"],
        "Retrieval":  ["retrieval", "search", "ranking", "recommendation"],
        "Embeddings": ["embedding", "embeddings", "sentence-transformer", "bge", "e5"],
        "Vector DB":  ["pinecone", "qdrant", "milvus", "weaviate", "faiss",
                       "opensearch", "elasticsearch", "vector search"],
        "Evaluation": ["ndcg", "mrr", "map", "offline benchmark",
                       "a/b", "ab test", "ragas", "deepeval", "trulens"],
    }
    weaknesses = [
        cap for cap, kws in required_capabilities.items()
        if not any(kw in profile_text for kw in kws)
    ]
    return strengths[:5], weaknesses[:4]


# ---------------------------------------------------------------------------
# Resume health
# ---------------------------------------------------------------------------

def calculate_resume_health(e, b) -> dict:
    """Calculate resume health metrics and overall risk."""
    health = {
        "timeline_valid":     e.years > 0 and len(e.companies) > 0,
        "profile_complete":   len(e.name) > 0 and len(e.title) > 0 and len(e.skills) > 1,
        "production_evidence": e.group_hits_career.get("production", 0) > 0,
        "expert_consistent":  b.penalty < 0.1,
        "keyword_stuffing":   len(e.skills) < 50,
    }
    health_score = sum(health.values()) / len(health)
    if health_score >= 0.8:
        health["overall_risk"]  = "Low"
        health["risk_color"]    = "#10b981"
    elif health_score >= 0.5:
        health["overall_risk"]  = "Medium"
        health["risk_color"]    = "#f59e0b"
    else:
        health["overall_risk"]  = "High"
        health["risk_color"]    = "#ef4444"
    return health


# ---------------------------------------------------------------------------
# Main card renderer
# ---------------------------------------------------------------------------

def generate_html_cards(scored) -> str:
    """Create pool banner and styled Candidate 360 cards for top 20 candidates."""
    if not scored:
        return (
            '<div style="padding:40px;text-align:center;color:#475569;">'
            '<div style="font-size:32px;margin-bottom:12px;">📭</div>'
            '<div style="font-size:15px;">No candidates evaluated yet.</div>'
            '<div style="font-size:12px;color:#334155;margin-top:6px;">'
            'Upload a dataset and run evaluation to see results.</div>'
            '</div>'
        )

    parts = [generate_pool_banner(scored)]

    for rank, (score, e, b) in enumerate(scored[:20], start=1):
        title          = esc(display_title(e.title) if e.title else "Technical Candidate")
        candidate_name = esc(e.name if e.name else "Unknown Candidate")
        company        = e.current_company.title() if e.current_company else "—"
        notice         = int(e.signals.get("notice_period_days") or 0)
        resp           = float(e.signals.get("recruiter_response_rate") or 0.0) * 100
        location       = esc(e.location.title() if e.location else "India")
        last_active    = esc(e.signals.get("last_active_date", "N/A"))
        penalty_flag   = b.penalty > 0.05
        open_to_work   = e.signals.get("open_to_work_flag", False)

        confidence  = calculate_confidence_score(e, b, score)
        conf_color  = get_confidence_color(confidence)
        conf_label  = get_confidence_label(confidence)
        strengths, weaknesses = get_strengths_weaknesses(e)
        health      = calculate_resume_health(e, b)

        bars             = _signal_progress_bars(b)
        evidence_checklist = _evidence_checklist(e)

        # Score color band
        score_val = score * 10
        if score_val >= 7:
            score_color = "#10b981"
        elif score_val >= 5:
            score_color = "#f59e0b"
        else:
            score_color = "#94a3b8"

        # Rank badge color
        rank_colors = {1: "#fbbf24", 2: "#94a3b8", 3: "#cd7f32"}
        rank_color  = rank_colors.get(rank, "#6366f1")

        penalty_row = (
            '<div style="margin:8px 0;padding:8px 12px;background:rgba(251,146,60,0.08);'
            'border-radius:6px;border-left:2px solid #fb923c;display:flex;align-items:center;gap:8px;">'
            '<span style="color:#fb923c;font-size:11px;font-weight:600;">⚠ Audit penalty applied</span>'
            '</div>'
            if penalty_flag else ""
        )

        otw_badge = (
            '<span style="display:inline-flex;align-items:center;gap:4px;'
            'background:rgba(16,185,129,0.15);color:#6ee7b7;'
            'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;'
            'border:1px solid rgba(16,185,129,0.3);">● Open to Work</span>'
            if open_to_work else ""
        )

        confidence_badge = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:rgba({_hex_to_rgb(conf_color)},0.12);'
            f'border:1px solid rgba({_hex_to_rgb(conf_color)},0.35);'
            f'color:{conf_color};padding:5px 12px;border-radius:20px;font-size:11px;font-weight:700;">'
            f'🎯 {confidence}% · {conf_label} Confidence</div>'
        )

        # Strengths
        strengths_html = ""
        if strengths:
            badges = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'background:rgba(16,185,129,0.1);color:#6ee7b7;'
                f'padding:3px 9px;border-radius:4px;font-size:11px;margin:2px 3px 2px 0;'
                f'border:1px solid rgba(16,185,129,0.25);">✓ {s}</span>'
                for s in strengths
            )
            strengths_html = f'{_section_header("Strengths", "#10b981")}<div style="line-height:2;">{badges}</div>'

        # Weaknesses
        weaknesses_html = ""
        if weaknesses:
            badges = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'background:rgba(239,68,68,0.08);color:#fca5a5;'
                f'padding:3px 9px;border-radius:4px;font-size:11px;margin:2px 3px 2px 0;'
                f'border:1px solid rgba(239,68,68,0.2);">✗ {w}</span>'
                for w in weaknesses
            )
            weaknesses_html = f'{_section_header("Missing Skills", "#ef4444")}<div style="line-height:2;">{badges}</div>'

        # Resume health
        health_checks = [
            ("timeline_valid",      "Timeline"),
            ("profile_complete",    "Profile"),
            ("production_evidence", "Production"),
            ("expert_consistent",   "Claims"),
        ]
        health_items = []
        for key, label in health_checks:
            ok    = health[key]
            color = "#10b981" if ok else "#64748b"
            icon  = "✔" if ok else "✗"
            health_items.append(
                f'<span style="font-size:11px;color:{color};margin-right:14px;font-weight:{"600" if ok else "400"};">'
                f'{icon} {label}</span>'
            )
        risk_pill = (
            f'<span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;'
            f'background:rgba({_hex_to_rgb(health["risk_color"])},0.15);'
            f'color:{health["risk_color"]};border:1px solid rgba({_hex_to_rgb(health["risk_color"])},0.3);">'
            f'{health["overall_risk"]} Risk</span>'
        )
        health_html = (
            f'{_section_header("Resume Health", "#8b5cf6")}'
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
            f'{"".join(health_items)}{risk_pill}</div>'
        )

        summary = esc(reasoning(e, b))

        card = f"""
        <div style="border:1px solid #1e293b;border-radius:12px;padding:20px;margin-bottom:16px;
                    background:linear-gradient(135deg,#111827,#0d1623);
                    box-shadow:0 4px 20px rgba(0,0,0,0.3);
                    transition:box-shadow 0.2s ease;"
             onmouseover="this.style.boxShadow='0 6px 28px rgba(99,102,241,0.15)'"
             onmouseout="this.style.boxShadow='0 4px 20px rgba(0,0,0,0.3)'">

          <!-- Header row -->
          <div style="display:flex;justify-content:space-between;align-items:flex-start;
                      margin-bottom:10px;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              <span style="font-weight:800;color:{rank_color};font-size:18px;min-width:28px;">#{rank}</span>
              <div>
                <span style="font-weight:700;color:#f1f5f9;font-size:14px;">{candidate_name}</span>
                <span style="color:#475569;font-size:12px;margin:0 6px;">·</span>
                <span style="color:#64748b;font-size:12px;">{e.candidate_id}</span>
              </div>
              <span style="font-size:20px;font-weight:800;color:{score_color};">{score_val:.2f}<span style="font-size:12px;color:#64748b;font-weight:400;">/10</span></span>
              {otw_badge}
            </div>
            {confidence_badge}
          </div>

          <!-- Meta row -->
          <div style="display:flex;flex-wrap:wrap;gap:4px 0;margin:6px 0 14px;font-size:12px;">
            {''.join(f'<span style="color:#94a3b8;">{item}</span><span style="color:#1e293b;margin:0 10px;">|</span>' for item in [title, company, f"{e.years:.1f} yrs", location, f"Notice {notice}d", f"Response {resp:.0f}%", f"Active {last_active}"])[:-len('<span style="color:#1e293b;margin:0 10px;">|</span>')]}</div>

          {_section_header("Signal Scores")}
          {bars}
          {_section_header("Evidence")}
          {evidence_checklist}
          {strengths_html}
          {weaknesses_html}
          {health_html}
          {penalty_row}
          {_section_header("Summary", "#94a3b8")}
          <div style="font-size:12px;color:#cbd5e1;line-height:1.6;padding:10px 12px;
                      background:rgba(15,23,42,0.6);border-radius:6px;
                      border-left:2px solid #6366f155;">{summary}</div>
        </div>
        """
        parts.append(card)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' for use in rgba()."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "128,128,128"
