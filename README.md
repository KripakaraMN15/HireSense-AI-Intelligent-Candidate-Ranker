# 🎯 HireSense AI --- Intelligent Candidate Ranking System

> **Transparent, explainable, CPU-only candidate ranking platform**
> built for the **Redrob India Data & AI Challenge**.\
> HireSense AI evaluates candidate profiles using multiple
> recruiter-centric signals to identify the best matches for a Senior AI
> Engineer role while providing clear reasoning behind every ranking.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Plotly](https://img.shields.io/badge/Plotly-Analytics-blue) ![CPU
Only](https://img.shields.io/badge/CPU-Only-success) ![Explainable
AI](https://img.shields.io/badge/Explainable-AI-indigo)

------------------------------------------------------------------------

#  Features at a Glance

  -----------------------------------------------------------------------
  Feature                        Description
  ------------------------------ ----------------------------------------
   Multi-Signal Ranking        Scores candidates across six independent
                                 recruiter signals

   Executive Dashboard         High-level insights into the entire
                                 candidate pool

   Candidate 360 Cards         Rich recruiter-friendly candidate
                                 profiles

   Signal Evidence             Shows evidence supporting each
                                 recommendation

  Strength Analysis           Highlights candidate strengths relevant
                                 to the JD

   Resume Health               Detects profile quality and consistency

   Confidence Score            Confidence indicator for every
                                 recommendation

   Skill Gap Analysis          Visual comparison of candidate
                                 capability vs expected skills

   Interview Preparation       Automatically generated interview
                                 guidance

   CSV Export                  Export Top-100 ranked candidates
  -----------------------------------------------------------------------

------------------------------------------------------------------------

#  Table of Contents

-   Overview
-   Pipeline
-   Quick Start
-   How to Run
-   Dashboard Overview
-   Scoring Methodology
-   Ranking Signals
-   Project Structure
-   Output
-   Technology Stack
-   Future Improvements

------------------------------------------------------------------------

#  Pipeline

``` text
Candidate Dataset (.json/.jsonl)
            │
            ▼
      Build Candidate Evidence
            │
            ▼
 Multi-Signal Candidate Scoring
            │
            ▼
Consistency & Audit Checks
            │
            ▼
      Candidate Ranking
            │
            ▼
 ┌────────────────────────────┐
 │ Candidate 360 Cards        │
 │ Executive Summary          │
 │ Skill Gap Analysis         │
 │ Radar Comparison           │
 │ Interview Preparation      │
 │ CSV Export                 │
 └────────────────────────────┘
```

------------------------------------------------------------------------

# 🚀 Quick Start

``` bash
git clone <repository-url>
cd HireSense-AI

pip install -r requirements.txt

python app.py
```

Open:

    http://127.0.0.1:7860

------------------------------------------------------------------------

#  How to Use

1.  Launch the application.
2.  Upload a candidate dataset (`.json`, `.jsonl`, or `.jsonl.gz`).
3.  (Optional) Adjust the signal weight sliders.
4.  Click **Run Candidate Evaluation**.
5.  Review ranked candidates.
6.  Explore analytics and interview preparation.
7.  Export the Top-100 ranked candidates as CSV.

------------------------------------------------------------------------

#  Dashboard Overview

##  Candidate Shortlist

Displays recruiter-friendly candidate cards containing:

-   Overall ranking score
-   Candidate information
-   Six signal scores
-   Progress bars
-   Signal evidence checklist
-   Candidate strengths
-   Resume health indicators
-   Confidence score
-   Recruiter summary

------------------------------------------------------------------------

##  Executive Summary

Provides overall hiring insights including:

-   Pool size
-   Average score
-   Top score
-   Retrieval evidence
-   Open-to-work statistics
-   Average notice period
-   Score distribution

------------------------------------------------------------------------

##  Skill Gap Analysis

Visual comparison between expected capabilities and candidate strengths
across:

-   Python
-   Retrieval
-   Vector Databases
-   LLM
-   Evaluation
-   Production ML

------------------------------------------------------------------------

##  Candidate Comparison

Interactive radar visualization comparing multiple candidates across all
scoring dimensions.

------------------------------------------------------------------------

##  Interview Preparation

Generates recruiter-focused interview questions based on:

-   Production experience
-   Technical skills
-   Career evidence
-   Resume inconsistencies
-   Logistics

------------------------------------------------------------------------

##  CSV Export

Exports the Top-100 ranked candidates including:

-   Candidate ID
-   Rank
-   Score
-   Recruiter reasoning

------------------------------------------------------------------------

#  Scoring Methodology

HireSense AI evaluates every candidate using six independent signals.

  Signal          Purpose
  --------------- --------------------------------------------------
   Role Fit     Alignment between job title and target role
   Production   Production ML, retrieval and deployment evidence
   Career       Experience quality and progression
   Skills       Technical capability and credibility
   Logistics    Location, notice period and availability
   Behavior     Recruiter engagement and profile activity

The final score is computed from these signals and adjusted using
deterministic consistency checks.

------------------------------------------------------------------------

#  Explainable Ranking

Every recommendation is accompanied by:

-   Candidate summary
-   Signal evidence
-   Strength indicators
-   Confidence score
-   Resume health
-   Transparent reasoning

No external APIs or LLM inference are used during candidate ranking.

------------------------------------------------------------------------

#  Project Structure

``` text
HireSense-AI/
│
├── app.py
├── rank.py
├── ui/
│   └── cards.py
├── sample_candidates.json
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

#  Output

The application produces:

-   Ranked candidate shortlist
-   Executive dashboard
-   Candidate analytics
-   Skill gap charts
-   Radar comparison
-   Interview preparation
-   CSV export

------------------------------------------------------------------------

#  Technology Stack

  Technology   Purpose
  ------------ ---------------------------------
  Python       Core ranking engine
  Gradio       Interactive recruiter dashboard
  Plotly       Interactive visualizations
  Pandas       Data processing
  HTML/CSS     Candidate card rendering
  JSON         Candidate profile dataset

------------------------------------------------------------------------

#  Design Principles

-   Explainable AI
-   Transparent scoring
-   Recruiter-friendly interface
-   Modular architecture
-   Deterministic ranking
-   CPU-only execution

------------------------------------------------------------------------

#  Future Improvements

-   Job Description parser
-   Resume parser
-   Semantic profile matching
-   ATS integration
-   Recruiter feedback learning
-   Multi-role evaluation
-   Advanced analytics

------------------------------------------------------------------------

#  License

Developed for the **Redrob India Data & AI Challenge**.
