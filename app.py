"""
app.py  –  HireSense AI  |  Gradio application entry-point
============================================================
Orchestration only: evaluation pipeline, callbacks, Gradio layout.
Presentation helpers live in ui/cards.py.
"""

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is required.  pip install gradio")

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Plotly is required.  pip install plotly")

import io
import json
import re
import time
import gzip
import pandas as pd
from pathlib import Path

from rank import (
    build_evidence,
    score_candidate,
    reasoning,
    clamp,
    top_skill_names,
    display_title,
)

from ui.cards import (
    generate_html_cards,
    compute_pool_stats,          # still used by executive_summary below
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERVIEW_PREP_PLACEHOLDER = """
### Interview Prep

Run **Candidate Evaluation** first, then choose a shortlisted candidate from the dropdown.

Tailored questions are generated from the same evidence used for scoring — skills,
production signals, and audit flags.
"""

ALLOWED_EXTENSIONS      = {'.json', '.jsonl', '.gz'}
DEFAULT_SLIDER_WEIGHTS  = (0.29, 0.20, 0.16, 0.12, 0.09, 0.09)

# Skill gap chart constants (stay in app.py — used by plot_skill_gap_bars)
SKILL_REQUIRED_LEVELS = {
    "Python": 10, "Retrieval": 10, "Vector DB": 7,
    "LLM": 8, "Evaluation": 7, "Production": 9,
}
SKILL_COLORS = {
    "Python": "#3b82f6", "Retrieval": "#6366f1", "Vector DB": "#8b5cf6",
    "LLM": "#a855f7", "Evaluation": "#ec4899", "Production": "#10b981",
}


# ---------------------------------------------------------------------------
# Helpers (non-UI)
# ---------------------------------------------------------------------------

def load_candidates(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    if path.endswith((".jsonl", ".jsonl.gz")):
        with opener(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def using_default_weights(weights) -> bool:
    return all(abs(float(a) - b) < 1e-9 for a, b in zip(weights, DEFAULT_SLIDER_WEIGHTS))


def format_stat_row(scored, stats):
    return str(stats.get("count", 0)), stats.get("time_ms", "0 ms"), stats.get("top", "0.00/10")


def extract_candidate_id(selection: str | None) -> str | None:
    if not selection:
        return None
    match = re.search(r"CAND_\d{7}", selection)
    return match.group(0) if match else selection.strip() or None


def candidate_prep_choices(scored, limit: int = 20) -> list[str]:
    choices = []
    for rank, (_, e, _) in enumerate(scored[:limit], start=1):
        title = display_title(e.title) if e.title else "Unknown"
        choices.append(f"#{rank} {e.candidate_id} — {title}")
    return choices


def reset_weights():
    return 0.29, 0.20, 0.16, 0.12, 0.09, 0.09


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(file_obj, w_role, w_prod, w_career, w_skills, w_logistics, w_behavior):
    if file_obj is None:
        return [], {"count": 0, "time_ms": 0, "top": "0.00/10"}

    ext = Path(file_obj.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return [], {"error": f"⚠️ Invalid file type `{ext}`. Upload a `.json` or `.jsonl` file."}

    start_t = time.time()
    try:
        data = load_candidates(file_obj.name)
    except Exception as exc:
        return [], {"error": f"⚠️ Error parsing file: `{exc}`"}

    candidates = data if isinstance(data, list) else [data]
    if not candidates:
        return [], {"error": "⚠️ Uploaded file contains no valid profiles."}

    weights        = (w_role, w_prod, w_career, w_skills, w_logistics, w_behavior)
    use_ranker     = using_default_weights(weights)
    total_w        = sum(weights)
    if total_w > 0:
        w_r, w_p, w_c, w_s, w_l, w_b = [w / total_w for w in weights]
    else:
        w_r, w_p, w_c, w_s, w_l, w_b = DEFAULT_SLIDER_WEIGHTS

    scored = []
    for c in candidates:
        e      = build_evidence(c)
        b_orig = score_candidate(e)
        if use_ranker:
            final = b_orig.final
        else:
            raw   = (w_r * b_orig.role + w_p * b_orig.production + w_c * b_orig.career +
                     w_s * b_orig.skills + w_l * b_orig.logistics + w_b * b_orig.behavior)
            final = clamp(raw - b_orig.penalty)
        scored.append((final, e, b_orig))

    scored.sort(key=lambda x: (-x[0], x[1].candidate_id))
    elapsed_ms = (time.time() - start_t) * 1000
    stats = {
        "count":   len(scored),
        "time_ms": f"{elapsed_ms:.1f} ms",
        "top":     f"{scored[0][0] * 10:.2f}/10" if scored else "0.00/10",
    }
    return scored, stats


def clear_state():
    return [], {"count": 0, "time_ms": "0 ms", "top": "0.00/10"}


# ---------------------------------------------------------------------------
# Executive summary (stays in app.py — Phase 2 territory)
# ---------------------------------------------------------------------------

def compute_candidate_skill_values(e, b):
    from ui.cards import skill_name_set
    skill_names = skill_name_set(e)
    return {
        "Python":     min(10, e.group_hits_career.get("python", 0) * 3 + e.group_hits_all.get("python", 0)),
        "Retrieval":  min(10, e.group_hits_career.get("retrieval", 0) * 3),
        "Vector DB":  8 if any("vector" in n for n in skill_names) else (2 if any("db" in n for n in skill_names) else 0),
        "LLM":        min(10, e.group_hits_career.get("llm_nlp", 0) * 3),
        "Evaluation": min(10, e.group_hits_career.get("evaluation", 0) * 3),
        "Production": min(10, int(b.production * 10)),
    }


def executive_summary(scored):
    if not scored:
        return "<p style='color:#64748b;padding:24px;text-align:center;'>No candidates evaluated.</p>"

    metrics = compute_pool_stats(scored)
    if not metrics:
        return "<p style='color:#64748b;'>Could not compute pool statistics.</p>"

    n               = metrics['count']
    avg_score       = metrics['avg_score']
    top_score       = max(s for s, _, _ in scored) * 10
    retrieval_count = metrics['retrieval_evidence']
    otw_count       = metrics['open_to_work']
    avg_notice      = metrics['avg_notice']
    top_gap         = metrics['top_skill_gap']

    score_values = [s * 10 for s, _, _ in scored]
    score_bands  = {"0-3": 0, "3-5": 0, "5-7": 0, "7-9": 0, "9-10": 0}
    for v in score_values:
        if v < 3:   score_bands["0-3"]  += 1
        elif v < 5: score_bands["3-5"]  += 1
        elif v < 7: score_bands["5-7"]  += 1
        elif v < 9: score_bands["7-9"]  += 1
        else:       score_bands["9-10"] += 1

    band_colors = {
        "0-3": "#ef4444", "3-5": "#fb923c",
        "5-7": "#eab308", "7-9": "#22c55e", "9-10": "#10b981",
    }

    stat_cards = [
        ("Pool Size",          str(n),               "candidates",    "#ffffff"),
        ("Top Score",          f"{top_score:.2f}",   "out of 10",     "#6366f1"),
        ("Avg Score",          f"{avg_score:.2f}",   "out of 10",     "#22c55e"),
        ("Retrieval Evidence", str(retrieval_count), f"{(retrieval_count/max(n,1))*100:.0f}% of pool", "#3b82f6"),
        ("Open to Work",       str(otw_count),       f"{(otw_count/max(n,1))*100:.0f}% of pool",      "#6ee7b7"),
        ("Avg Notice",         f"{avg_notice:.0f}",  "days",          "#f59e0b"),
    ]

    cards_html = "".join(
        f'<div style="background:linear-gradient(135deg,#1e293b,#111827);'
        f'border:1px solid #1e293b;border-radius:10px;padding:18px;text-align:center;">'
        f'<div style="font-size:10px;color:#475569;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1;">{value}</div>'
        f'<div style="font-size:10px;color:#64748b;margin-top:4px;">{sub}</div></div>'
        for label, value, sub, color in stat_cards
    )

    dist_bars_html = "".join(
        f'<div style="margin-bottom:6px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;'
        f'color:#94a3b8;margin-bottom:3px;"><span>{band}</span><span style="font-weight:600;">{count}</span></div>'
        f'<div style="background:#0f172a;border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="background:{band_colors[band]};width:{(count/max(n,1))*100:.0f}%;'
        f'height:100%;border-radius:4px;box-shadow:0 0 4px {band_colors[band]}66;"></div></div></div>'
        for band, count in score_bands.items()
    )

    return f"""
    <div style="padding:4px;">
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                  gap:10px;margin-bottom:16px;">
        {cards_html}
      </div>
      <div style="background:linear-gradient(135deg,#1e293b,#111827);
                  border:1px solid #1e293b;border-radius:10px;padding:18px;">
        <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.8px;margin-bottom:12px;">Score Distribution</div>
        {dist_bars_html}
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1e293b;">
          <span style="font-size:11px;color:#64748b;">Top skill gap across pool: </span>
          <span style="font-size:12px;font-weight:700;color:#fb923c;">{top_gap}</span>
        </div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Skill gap chart
# ---------------------------------------------------------------------------

def plot_skill_gap_bars(scored, candidate_index: int = 0):
    if not scored or candidate_index >= len(scored):
        return go.Figure()

    _, e, b = scored[candidate_index]
    candidate_vals = compute_candidate_skill_values(e, b)
    skills   = list(SKILL_REQUIRED_LEVELS.keys())
    required = [SKILL_REQUIRED_LEVELS[s] for s in skills]
    candidate = [candidate_vals[s] for s in skills]
    gaps      = [max(0, required[i] - candidate[i]) for i in range(len(skills))]
    colors    = [SKILL_COLORS[s] for s in skills]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Required", y=skills, x=required, orientation='h',
        marker=dict(color='#334155', line=dict(color='#475569', width=1)),
        text=[str(v) for v in required], textposition='outside',
        textfont=dict(color='#94a3b8', size=10),
        hovertemplate='Required: %{x}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name="Candidate", y=skills, x=candidate, orientation='h',
        marker=dict(color=colors, line=dict(color='#ffffff', width=0.5)),
        text=[str(v) for v in candidate], textposition='inside',
        textfont=dict(color='#ffffff', size=11),
        hovertemplate='Candidate: %{x}<extra></extra>',
    ))

    annotations = [
        dict(x=max(required[i], candidate[i]) + 0.3, y=i, text=f"Gap {g}",
             showarrow=False, font=dict(color='#fb923c', size=10),
             xanchor='left')
        for i, (_, g) in enumerate(zip(skills, gaps)) if g > 0
    ]

    fig.update_layout(
        title=dict(text=f"Skill Gap — {e.candidate_id}",
                   font=dict(color='#e2e8f0', size=13), x=0.5),
        barmode='group', height=340,
        margin=dict(l=10, r=80, t=48, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', size=11),
        xaxis=dict(title="Proficiency (0–10)", range=[0, 14],
                   showgrid=True, gridcolor='#1e293b', tickfont=dict(color='#64748b')),
        yaxis=dict(tickfont=dict(color='#e2e8f0', size=12), gridcolor='#1e293b'),
        legend=dict(font=dict(color='#94a3b8', size=11), orientation='h',
                    y=1.08, x=0.5, xanchor='center'),
        hovermode='y unified', annotations=annotations,
    )
    return fig


def _skill_gap_from_selection(scored, selection):
    if not scored or not selection:
        return go.Figure()
    cid = extract_candidate_id(selection)
    if not cid:
        return plot_skill_gap_bars(scored, 0)
    for idx, (_, e, _) in enumerate(scored):
        if e.candidate_id == cid:
            return plot_skill_gap_bars(scored, idx)
    return plot_skill_gap_bars(scored, 0)


def _executive_from_scored(scored):
    return executive_summary(scored) if scored else "<p style='color:#64748b;padding:24px;'>No candidates evaluated.</p>"


# ---------------------------------------------------------------------------
# Interview prep
# ---------------------------------------------------------------------------

def interview_prep(scored, selection):
    if not scored:
        return INTERVIEW_PREP_PLACEHOLDER

    candidate_id = extract_candidate_id(selection)
    if not candidate_id:
        return "*Select a candidate from the dropdown above.*"

    for rank, (score, e, b) in enumerate(scored, start=1):
        if e.candidate_id != candidate_id:
            continue

        title      = display_title(e.title) if e.title else "Candidate"
        skills     = top_skill_names(e, limit=6)
        skill_line = ", ".join(skills) if skills else "general ML / Python stack"
        notice     = int(e.signals.get("notice_period_days") or 0)

        questions = []
        if b.production >= 0.5:
            questions.append(
                "Walk me through a production retrieval or ranking system you shipped — "
                "latency, monitoring, and how you measured quality (NDCG, MRR, recall@k)."
            )
        else:
            questions.append(
                "Describe your most hands-on experience building search, recommendation, "
                "or RAG pipelines in production."
            )

        if e.group_hits_career.get("evaluation", 0):
            questions.append(
                "How did you design offline or online evaluation for your ranking/retrieval models? "
                "Share a concrete benchmark or A/B test."
            )
        if skills:
            questions.append(
                f"You list **{skills[0]}** — how long have you used it in production, "
                "and what was the hardest scaling or quality issue you solved?"
            )
        if b.penalty >= 0.1:
            questions.append(
                "Your profile shows mixed signals between listed skills and career history — "
                "can you point to specific projects where you personally owned the ML/retrieval work?"
            )
        if b.logistics < 0.5 or notice >= 60:
            questions.append(
                f"This role targets Tier-1 India onsite/hybrid — discuss relocation, "
                f"notice period ({notice} days), and your earliest start date."
            )

        probe_areas = []
        if b.role < 0.6:
            probe_areas.append("title/role fit vs. Senior AI Engineer JD")
        if b.production < 0.4:
            probe_areas.append("limited production systems evidence")
        if b.career < 0.5:
            probe_areas.append("career progression and product-company exposure")

        concerns = []
        if b.penalty >= 0.15:
            concerns.append("consistency/trap penalty applied during scoring")
        if e.suspicious_skill_count:
            concerns.append("expert skills with very short declared duration")
        if float(e.signals.get("recruiter_response_rate") or 0.0) < 0.2:
            concerns.append("low recruiter response rate on platform")

        lines = [
            f"## Interview Prep — #{rank} {e.candidate_id}",
            f"**{title}** · {e.years:.1f} yrs · Score **{score * 10:.2f}/10**",
            "",
            f"**Profile summary:** {reasoning(e, b)}",
            "",
            "### Highlighted skills to validate",
            skill_line,
            "",
            "### Suggested questions",
        ]
        for i, q in enumerate(questions[:5], start=1):
            lines.append(f"{i}. {q}")

        if probe_areas:
            lines += ["", "### Areas to probe deeper", "- " + "\n- ".join(probe_areas)]
        if concerns:
            lines += ["", "### Audit flags to clarify", "- " + "\n- ".join(concerns)]

        return "\n".join(lines)

    return f"**Candidate ID `{candidate_id}` not found** in the current evaluation pool."


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(scored, count: int = 100) -> str:
    if not scored:
        return ""
    count = max(1, min(count, len(scored)))
    rows  = []
    for rank, (score, e, b) in enumerate(scored[:count], start=1):
        adjusted = max(0.0, score - (rank - 1) * 1e-7)
        rows.append({
            "candidate_id": e.candidate_id,
            "rank":         rank,
            "score":        f"{adjusted:.6f}",
            "reasoning":    reasoning(e, b),
        })
    buf = io.StringIO()
    pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"]).to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Custom CSS — refined dark theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Base ───────────────────────────────────────────── */
body, .gradio-container {
    background: #080e1a !important;
    font-family: 'Inter', 'DM Sans', system-ui, sans-serif !important;
}

/* ── Tabs ───────────────────────────────────────────── */
.tabs > .tab-nav {
    background: #0d1623 !important;
    border-bottom: 1px solid #1e293b !important;
    gap: 2px !important;
}
.tabs > .tab-nav button {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
    transition: color 0.2s, background 0.2s !important;
}
.tabs > .tab-nav button.selected {
    color: #a5b4fc !important;
    background: #1e293b !important;
    border-bottom: 2px solid #6366f1 !important;
}
.tabs > .tab-nav button:hover:not(.selected) {
    color: #94a3b8 !important;
    background: #111827 !important;
}

/* ── Sliders ────────────────────────────────────────── */
.gradio-slider input[type="range"] {
    accent-color: #6366f1 !important;
}

/* ── Primary buttons ─────────────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.35) !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover {
    opacity: 0.9 !important;
}

/* ── Secondary buttons ───────────────────────────────── */
.gr-button-secondary {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.gr-button-secondary:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
}

/* ── Textboxes / inputs ──────────────────────────────── */
input[type="text"], .gr-textbox textarea {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
}
input[type="text"]:focus, .gr-textbox textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

/* ── Labels ──────────────────────────────────────────── */
label.svelte-1b6s6vi, .gr-label {
    color: #6366f1 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* ── Stat boxes ──────────────────────────────────────── */
.stat-box input {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    text-align: center !important;
}

/* ── File upload area ────────────────────────────────── */
.gr-file-upload {
    border: 1px dashed #334155 !important;
    border-radius: 10px !important;
    background: #0d1623 !important;
    transition: border-color 0.2s !important;
}
.gr-file-upload:hover {
    border-color: #6366f1 !important;
}

/* ── Dropdown ────────────────────────────────────────── */
.gr-dropdown select, .gr-dropdown .wrap {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Markdown ─────────────────────────────────────────── */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #f1f5f9 !important;
}
.gr-markdown p, .gr-markdown li {
    color: #94a3b8 !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
}
.gr-markdown strong { color: #e2e8f0 !important; }
.gr-markdown code {
    background: #1e293b !important;
    color: #a5b4fc !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 12px !important;
}

/* ── Panels / columns ────────────────────────────────── */
.gr-panel, .gr-box {
    background: #0d1623 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080e1a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
"""


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="HireSense AI",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown(
        "# 🎯 HireSense AI\n"
        "### *Intelligent Candidate Ranker · Senior AI Engineer · Tier-1 India*"
    )

    with gr.Row(equal_height=False):

        # ── Left panel: JD rubric + weight sliders ───────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown(
                "**Role** Senior AI Engineer\n\n"
                "**Location** Tier-1 India\n\n"
                "**Experience** 5–9 years"
            )
            gr.Markdown("---")
            gr.Markdown("#### 🎛 Signal Weights")

            slider_role      = gr.Slider(0.0, 0.5, value=0.29, step=0.01, label="Role & Title Fit")
            slider_prod      = gr.Slider(0.0, 0.5, value=0.20, step=0.01, label="Production Systems")
            slider_career    = gr.Slider(0.0, 0.5, value=0.16, step=0.01, label="Career History")
            slider_skills    = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Skills Credibility")
            slider_logistics = gr.Slider(0.0, 0.5, value=0.09, step=0.01, label="Logistics & Location")
            slider_behavior  = gr.Slider(0.0, 0.5, value=0.09, step=0.01, label="Platform Engagement")

            gr.Markdown("---")
            with gr.Row():
                btn_apply = gr.Button("🚀 Apply Weights", variant="primary", scale=3)
                btn_reset = gr.Button("↺ Reset",          variant="secondary", scale=1)

        # ── Right panel: stats + file + tabs ────────────────────────────
        with gr.Column(scale=3):

            # Stat row
            with gr.Row():
                stat_count = gr.Textbox(label="Pool Ranked",  value="0",       interactive=False, elem_classes=["stat-box"])
                stat_time  = gr.Textbox(label="Re-Rank Time", value="0 ms",    interactive=False, elem_classes=["stat-box"])
                stat_top   = gr.Textbox(label="Top Score",    value="0.00/10", interactive=False, elem_classes=["stat-box"])

            file_input = gr.File(label="Upload Candidate Dataset (.jsonl or .json)")
            btn_run    = gr.Button("🚀 Run Candidate Evaluation", variant="primary")

            with gr.Tabs():

                with gr.TabItem("🏆 Candidate Shortlist"):
                    out_html = gr.HTML()

                with gr.TabItem("📊 Executive Summary"):
                    out_exec = gr.HTML()

                with gr.TabItem("📊 Skill Gap Analysis"):
                    gr.Markdown(
                        "Compare a candidate's skills against the Senior AI Engineer JD requirements."
                    )
                    skill_gap_selector = gr.Dropdown(
                        label="Candidate", choices=[], value=None, interactive=True, scale=3,
                    )
                    out_skill_gap = gr.Plot()

                with gr.TabItem("🗣️ Interview Prep"):
                    gr.Markdown(
                        "Generate tailored questions from a candidate's score profile, skills, and audit flags."
                    )
                    interview_selector = gr.Dropdown(
                        label="Candidate (top 20)", choices=[], value=None, interactive=True,
                    )
                    out_prep = gr.Markdown(value=INTERVIEW_PREP_PLACEHOLDER)

                with gr.TabItem("📥 Export CSV"):
                    with gr.Row():
                        export_count = gr.Number(
                            label="Number of Candidates to Export",
                            value=100, precision=0, minimum=1,
                        )
                        btn_export = gr.Button("Export to CSV", variant="secondary")
                    out_csv = gr.File(label="CSV Download")

                with gr.TabItem("🛡️ Methodology"):
                    gr.Markdown("""
### 🛡️ Trap Auditing Rules
- **Fake Expert Skills** — flags expert proficiency claims with ≤3 months duration.
- **Services-Only Careers** — down-weights careers spent entirely in consulting without product exposure.
- **Non-Technical AI Claims** — penalises non-technical titles claiming AI keywords.

### 📜 Scoring Model
Bounded feature weighting across six dimensions (Role, Production, Career, Skills, Logistics, Behavior),
subtracted by a consistency penalty. Default weights match the internal `rank.py` ranker;
adjust sliders to re-rank with custom priorities.
                    """)

    state_scored = gr.State([])

    # ── Main callback ────────────────────────────────────────────────────────

    def _run(file_obj, w1, w2, w3, w4, w5, w6):
        scored, stats   = run_evaluation(file_obj, w1, w2, w3, w4, w5, w6)
        stat_vals       = format_stat_row(scored, stats)
        html_cards      = generate_html_cards(scored)
        exec_html       = _executive_from_scored(scored)
        prep_choices    = candidate_prep_choices(scored)
        first           = prep_choices[0] if prep_choices else None
        prep_md         = interview_prep(scored, first) if first else INTERVIEW_PREP_PLACEHOLDER
        skill_chart     = _skill_gap_from_selection(scored, first) if first else go.Figure()
        return (
            scored,
            stat_vals[0], stat_vals[1], stat_vals[2],
            html_cards,
            exec_html,
            gr.update(choices=prep_choices, value=first),
            skill_chart,
            gr.update(choices=prep_choices, value=first),
            prep_md,
        )

    _run_outputs = [
        state_scored, stat_count, stat_time, stat_top,
        out_html, out_exec,
        skill_gap_selector, out_skill_gap,
        interview_selector, out_prep,
    ]
    _run_inputs = [
        file_input,
        slider_role, slider_prod, slider_career,
        slider_skills, slider_logistics, slider_behavior,
    ]

    btn_run.click(fn=_run, inputs=_run_inputs, outputs=_run_outputs)
    btn_apply.click(fn=_run, inputs=_run_inputs, outputs=_run_outputs)

    btn_reset.click(
        fn=reset_weights,
        outputs=[slider_role, slider_prod, slider_career, slider_skills, slider_logistics, slider_behavior],
    )

    file_input.clear(
        fn=lambda: (
            [], "0", "0 ms", "0.00/10", "",
            "<p style='color:#64748b;padding:24px;'>No candidates evaluated.</p>",
            gr.update(choices=[], value=None), go.Figure(),
            gr.update(choices=[], value=None),
            INTERVIEW_PREP_PLACEHOLDER, None,
        ),
        inputs=[],
        outputs=[
            state_scored, stat_count, stat_time, stat_top,
            out_html, out_exec,
            skill_gap_selector, out_skill_gap,
            interview_selector, out_prep, out_csv,
        ],
    )

    skill_gap_selector.change(
        fn=_skill_gap_from_selection,
        inputs=[state_scored, skill_gap_selector],
        outputs=out_skill_gap,
    )

    interview_selector.change(
        fn=interview_prep,
        inputs=[state_scored, interview_selector],
        outputs=out_prep,
    )

    def _export(scored, count):
        import os, tempfile
        count    = int(count) if count else 100
        csv_text = export_csv(scored, count)
        if not csv_text:
            return None
        filename  = f"hiresense_top{count}.csv"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        return temp_path

    btn_export.click(fn=_export, inputs=[state_scored, export_count], outputs=out_csv)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())
