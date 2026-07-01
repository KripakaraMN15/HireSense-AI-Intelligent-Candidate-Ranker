---
app_file: app.py
colorFrom: blue
colorTo: indigo
emoji: 🎯
pinned: false
sdk: gradio
sdk_version: 6.19.0
title: HireSense AI Intelligent Candidate Ranker
---

# 🎯 HireSense AI -- Intelligent Candidate Ranker

HireSense AI is a transparent, explainable candidate ranking platform
built for the **Redrob India Data & AI Challenge**. It evaluates
candidate profiles using deterministic scoring instead of black-box AI,
producing recruiter-friendly rankings, evidence, interview guidance, and
analytics.

Unlike keyword matching systems, HireSense AI combines **career
evidence, production experience, skills, logistics, recruiter signals,
and behavioral indicators** to estimate candidate-job fit while applying
consistency checks to reduce inflated or misleading profiles.

------------------------------------------------------------------------

# ✨ Key Features

-   Transparent multi-signal candidate ranking
-   Six independent scoring dimensions
-   Candidate 360° profile cards with explanations
-   Signal evidence checklist
-   Strength analysis
-   Resume health indicators
-   Confidence score for every recommendation
-   Executive dashboard with pool analytics
-   Skill gap visualization
-   Radar comparison between candidates
-   Tailored interview preparation
-   Export Top-100 candidates to CSV
-   Fully CPU-based (no GPU required)
-   No external APIs or LLM inference during ranking

------------------------------------------------------------------------

# 🧠 Scoring Signals

Each candidate is evaluated across six dimensions:

  -----------------------------------------------------------------------
  Signal                        Description
  ----------------------------- -----------------------------------------
  Role                          Title and role alignment with the job
                                description

  Production                    Production ML, retrieval, search and
                                deployment evidence

  Career                        Experience quality, progression and
                                company background

  Skills                        Verified technical skills weighted by
                                credibility

  Logistics                     Location, notice period, work preference
                                and availability

  Behavior                      Recruiter engagement and profile activity
  -----------------------------------------------------------------------

These signals are combined into a final ranking score while applying
audit penalties for inconsistent or suspicious profiles.

------------------------------------------------------------------------

# 🛡️ Profile Auditing

The ranking engine detects quality issues such as:

-   AI keyword stuffing
-   Unsupported skill claims
-   Services-only experience with little production evidence
-   Timeline inconsistencies
-   Unrealistic expert proficiency
-   Weak career evidence compared to claimed skills

This improves ranking quality while keeping the reasoning fully
explainable.

------------------------------------------------------------------------

# 📊 Dashboard

The Gradio application includes:

-   Candidate Shortlist
-   Executive Summary
-   Candidate Analytics
-   Skill Gap Analysis
-   Radar Comparison
-   Interview Preparation
-   CSV Export
-   Audit Methodology

------------------------------------------------------------------------

# 🚀 How to Run

## 1. Install

``` bash
pip install -r requirements.txt
```

## 2. Launch the application

``` bash
python app.py
```

Open:

    http://127.0.0.1:7860

------------------------------------------------------------------------

# 📁 Supported Input

Upload candidate data as:

-   JSON
-   JSONL
-   JSONL.GZ

A sample dataset is included for testing.

------------------------------------------------------------------------

# 📖 How to Use

1.  Launch the application.
2.  Upload a candidate dataset.
3.  Adjust signal weights (optional).
4.  Run Candidate Evaluation.
5.  Review ranked candidates and evidence.
6.  Explore analytics, radar charts, interview preparation and skill
    gaps.
7.  Export the ranked Top-100 candidates.

------------------------------------------------------------------------

# 📤 Output

The application generates:

-   Ranked candidate list
-   Candidate reasoning
-   Signal scores
-   Evidence checklist
-   Resume health indicators
-   Skill gap visualization
-   Executive summary
-   Interview preparation guide
-   CSV export

------------------------------------------------------------------------

# 🏗️ Project Structure

    app.py               # Gradio application
    rank.py              # Scoring and ranking engine
    ui/
        cards.py         # Candidate card rendering
    sample_candidates.json
    requirements.txt
    README.md

------------------------------------------------------------------------

# ⚙️ Technologies

-   Python
-   Gradio
-   Plotly
-   Pandas
-   HTML/CSS
-   JSON

------------------------------------------------------------------------

# 🎯 Design Principles

-   Explainable AI
-   Transparent scoring
-   Deterministic ranking
-   Recruiter-friendly interface
-   Modular architecture
-   Fast CPU execution

------------------------------------------------------------------------

# 📈 Future Improvements

-   Resume parsing
-   JD parsing
-   Semantic embeddings
-   LLM-powered explanations
-   Multi-JD comparison
-   ATS integration
-   Real-time candidate search
-   Recruiter feedback learning

------------------------------------------------------------------------

# 📄 License

Created for the Redrob India Data & AI Challenge.
