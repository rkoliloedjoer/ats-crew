# tests/test_alignment.py
from src.tools.scoring import compute_metrics
from src.common.helpers import _augment_synonyms
from src.api.main import (
    filter_bullets_for_relevance,
    backfill_role_bullets,
    propose_jd_promotions,
)

def test_compute_metrics_canonical_variants():
    jd_phrases = ["cross-functional teams"]
    jd_atomic = ["power bi", "sql pipelines"]
    syns = {"sql pipelines": ["etl pipelines"]}
    pv = _augment_synonyms(jd_phrases, {})         # phrase variants map
    av = _augment_synonyms(jd_atomic, syns)        # atomic variants map

    text = """
    SUMMARY
    Built dashboards for cross functional teams.

    EXPERIENCE
    - Automated ETL pipeline jobs and Power-BI reports.
    """

    phr = compute_metrics(text, jd_phrases, pv, density_target=1)
    atm = compute_metrics(text, jd_atomic, av, density_target=2)

    assert phr["coverage"] == 1.0                      # hyphen/space normalized
    # power bi (1), sql/etl (1): density_target=2 ⇒ low density item present
    low = {x["term"] for x in atm["low_density_terms"]}
    assert "power bi" in low or "sql pipelines" in low
    assert atm["coverage"] >= 0.5

def test_filter_relevance_basic():
    bullets = [
        "Built Power-BI sales dashboards",
        "Managed stakeholder communications",
        "Wrote SQL pipelines for ETL orchestration",
    ]
    jd_atomic = ["power bi", "sql pipelines"]
    syns = {"sql pipelines": ["etl pipelines"]}
    keep = filter_bullets_for_relevance(
        bullets, jd_atomic, syns, min_matches=1, max_keep=8
    )
    assert any("Power-BI" in b or "Power" in b for b in keep)
    assert any("SQL pipelines" in b for b in keep)  # exact atom (or variant) is required
    assert "Managed stakeholder communications" not in keep

def test_backfill_role_bullets(tmp_path):
    # Create a tiny bank with one role that has 1 selected bullet + 2 candidates
    from tests.conftest import mk_bank
    bank = mk_bank(tmp_path, [
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Built Power-BI dashboards","recency_score":0.9,"impact_score":0.8,"leadership_score":0.4},
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Owned SQL pipeline orchestration","recency_score":0.8,"impact_score":0.7,"leadership_score":0.5},
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Presented insights to execs","recency_score":0.7,"impact_score":0.6,"leadership_score":0.6},
    ])
    selected = ["Built Power-BI dashboards"]
    jd_atomic = ["power bi", "sql pipelines"]
    syns = {"sql pipelines": ["etl pipelines"]}

    out = backfill_role_bullets(
        selected_bullets=selected,
        jd_atomic=jd_atomic,
        atomic_synonyms=syns,
        bank_path=bank,
        per_role_min=3,
        top_roles=1,
        max_total=4
    )
    assert len(out) >= 3                               # got backfilled
    assert any("SQL" in b or "pipeline" in b for b in out)

def test_promotions(tmp_path):
    from tests.conftest import mk_bank
    bank = mk_bank(tmp_path, [
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Built Power-BI dashboards","recency_score":0.9,"impact_score":0.8,"leadership_score":0.4},
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Owned SQL pipeline orchestration","recency_score":0.8,"impact_score":0.7,"leadership_score":0.5},
        {"role_title":"Mgr","company":"A","location":"HK","start_yyyy_mm":"2023-01","end_yyyy_mm":"2024-12",
         "bullet_text":"Presented insights to execs","recency_score":0.7,"impact_score":0.6,"leadership_score":0.6},
    ])
    # alignment_report stub: “sql pipelines” is currently missing (density=0)
    alignment_report = {
        "targets":{"density_target_atomic":2},
        "atomic":{
            "term_results":[
                {"term":"power bi","density":1,"in_experience":True},
                {"term":"sql pipelines","density":0,"in_experience":False},
            ]
        }
    }
    selected = ["Built Power-BI dashboards"]           # current resume bullets
    jd_atomic = ["power bi", "sql pipelines"]
    syns = {"sql pipelines": ["etl pipelines"]}

    promos = propose_jd_promotions(
        alignment_report=alignment_report,
        selected_bullets=selected,
        jd_atomic=jd_atomic,
        atomic_synonyms=syns,
        bank_path=bank,
        max_promotions=2
    )
    assert promos, "Expected at least one promotion suggestion"
    assert any("sql" in p["to_bullet"].lower() or "etl" in p["to_bullet"].lower() for p in promos)


# tests/test_alignment.py
import csv
from pathlib import Path

import pytest

# Imports from your project
from src.tools.scoring import compute_metrics
from src.api.main import (
    BANK_HEADERS,
    filter_bullets_for_relevance,
    backfill_role_bullets,
    propose_jd_promotions,
    build_canonical_variants,
)

# ---------------------------
# Helpers for temp bank setup
# ---------------------------
def write_bank(tmp_path: Path, rows: list[dict]) -> str:
    bank_path = tmp_path / "experience_bank.csv"
    with bank_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BANK_HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in BANK_HEADERS})
    return str(bank_path)


# ================
# compute_metrics()
# ================
def test_compute_metrics_variant_equivalence():
    """
    Hyphen/space + plural/singular variants should be treated consistently.
    """
    resume = (
        "HEADLINE\n"
        "Data Analyst\n\n"
        "SUMMARY\n"
        "Built analytics dashboards for leadership.\n\n"
        "EXPERIENCE\n"
        "- Delivered cross-functional teams analytics.\n"  # plural matches canonical
        "- Built dashboard automation.\n"
        "\nSKILLS\n"
        "sql, power bi\n"
        "\nCERTIFICATIONS\n"
        "—\n"
    )

    jd_phrases = ["cross-functional teams"]
    jd_atomic = ["dashboard"]  # singular in JD, plural/singular appear in resume
    jd_synonyms = {}  # none needed

    VAR = build_canonical_variants(jd_phrases, jd_atomic, jd_synonyms)

    # Phrases: density_target=1 (presence check)
    phr = compute_metrics(
        resume_text=resume,
        must_terms=jd_phrases,
        synonyms=VAR["phrase_map"],
        density_target=1,
    )
    assert phr["coverage"] == 1.0  # "cross-functional teams" ~ "cross functional team(s)"

    # Atomic: density_target=2 (we have "dashboards" (summary) + "dashboard" (experience))
    atm = compute_metrics(
        resume_text=resume,
        must_terms=jd_atomic,
        synonyms=VAR["atomic_map"],
        density_target=2,
    )
    assert atm["coverage"] == 1.0
    # Make sure the canonicalization prevents drift: density should hit target==2
    low = [x for x in atm["low_density_terms"] if x.get("term") == "dashboard"]
    assert low == []


# ==================================
# filter_bullets_for_relevance(...)
# ==================================
def test_filter_bullets_for_relevance_with_variant_index():
    bullets = [
        "Built cross functional team analytics",      # space
        "Built cross-functional team analytics",      # hyphen
        "Owned dashboard automation for execs",       # singular
        "Owned dashboards automation for execs",      # plural
        "General admin work",                         # irrelevant
    ]
    jd_atomic = ["cross-functional teams", "dashboard"]
    jd_synonyms = {}

    VAR = build_canonical_variants([], jd_atomic, jd_synonyms)

    kept = filter_bullets_for_relevance(
        bullets=bullets,
        jd_atomic=jd_atomic,
        synonyms=VAR["atomic_map"],
        min_matches=1,
        max_keep=8,
        variant_index=VAR["atomic_set"],  # <- canonical
    )

    # Irrelevant should be dropped
    assert "General admin work" not in kept
    # At least one of the cross functional (space/hyphen) and one of dashboard (sing/plural) should remain
    assert any("cross" in b.lower() for b in kept)
    assert any("dashboard" in b.lower() for b in kept)


# ==============================
# backfill_role_bullets(...) I
# ==============================
def test_backfill_role_bullets_top_role_backfilled(tmp_path):
    """
    If a selected role is short on bullets, we backfill from same role by JD match.
    """
    # Create a bank with two bullets in the same role, one selected, one candidate
    rows = [
        {
            "role_title": "Head of Analytics",
            "company": "ACME",
            "location": "HK",
            "start_yyyy_mm": "2022-01",
            "end_yyyy_mm": "2024-06",
            "bullet_text": "Owned dashboards automation for execs",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.8",
            "impact_score": "0.8",
            "leadership_score": "0.6",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        },
        {
            "role_title": "Head of Analytics",
            "company": "ACME",
            "location": "HK",
            "start_yyyy_mm": "2022-01",
            "end_yyyy_mm": "2024-06",
            "bullet_text": "Led cross-functional team analytics across regions",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.9",
            "impact_score": "0.7",
            "leadership_score": "0.7",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        },
    ]
    bank_path = write_bank(tmp_path, rows)

    selected = ["Owned dashboards automation for execs"]  # only 1 from that role
    jd_atomic = ["dashboard", "cross-functional teams"]
    jd_synonyms = {}
    VAR = build_canonical_variants([], jd_atomic, jd_synonyms)

    out = backfill_role_bullets(
        selected_bullets=selected,
        jd_atomic=jd_atomic,
        atomic_synonyms=VAR["atomic_map"],
        bank_path=bank_path,
        per_role_min=2,
        top_roles=1,
        max_total=8,
        variant_index=VAR["atomic_set"],  # <- canonical
    )
    assert len(out) >= 2
    # Ensure it pulled the cross-functional candidate from the same role
    assert any("cross" in b.lower() for b in out)


# =================================
# propose_jd_promotions(...) basics
# =================================
def test_propose_jd_promotions_suggests_swap(tmp_path):
    """
    Should propose swapping in a stronger bank bullet that hits a missing/low-density term.
    """
    # Bank has a strong bullet with the term "power bi"
    rows = [
        {
            "role_title": "Analytics Manager",
            "company": "Beta",
            "location": "HK",
            "start_yyyy_mm": "2021-01",
            "end_yyyy_mm": "2022-12",
            "bullet_text": "Built Power BI dashboards for executive reporting",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.7",
            "impact_score": "0.9",
            "leadership_score": "0.7",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        }
    ]
    bank_path = write_bank(tmp_path, rows)

    # Current selection lacks "power bi"
    selected = ["Owned dashboards automation for execs"]

    # Alignment says "power bi" is missing/low
    alignment_report = {
        "targets": {"density_target_atomic": 2},
        "atomic": {
            "term_results": [
                {"term": "power bi", "density": 0, "in_experience": True},
                {"term": "dashboard", "density": 1, "in_experience": True},
            ]
        },
    }

    jd_atomic = ["power bi", "dashboard"]
    jd_synonyms = {}
    VAR = build_canonical_variants([], jd_atomic, jd_synonyms)

    promos = propose_jd_promotions(
        alignment_report=alignment_report,
        selected_bullets=selected,
        jd_atomic=jd_atomic,
        atomic_synonyms=VAR["atomic_map"],
        bank_path=bank_path,
        max_promotions=3,
        variant_index=VAR["atomic_set"],  # <- canonical
    )
    assert promos, "Expected at least one promotion suggestion"
    # Sanity: the suggestion should introduce a bullet with 'power bi'
    assert any("power bi" in p.get("to_bullet", "").lower() for p in promos)


# ============================================
# Role-aware promotions used during /realign
# ============================================
def test_role_aware_promotions_prefer_same_role(tmp_path):
    """
    When multiple 'from_bullet' candidates exist, prefer swapping within the same role.
    """
    rows = [
        # Role A bullets
        {
            "role_title": "Analytics Manager",
            "company": "Gamma",
            "location": "HK",
            "start_yyyy_mm": "2023-01",
            "end_yyyy_mm": "2024-01",
            "bullet_text": "Owned dashboards automation for execs",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.8",
            "impact_score": "0.7",
            "leadership_score": "0.6",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        },
        {
            "role_title": "Analytics Manager",
            "company": "Gamma",
            "location": "HK",
            "start_yyyy_mm": "2023-01",
            "end_yyyy_mm": "2024-01",
            "bullet_text": "Built Power BI dashboards for executive reporting",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.8",
            "impact_score": "0.9",
            "leadership_score": "0.7",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        },
        # Role B bullet (should be less preferred)
        {
            "role_title": "BI Developer",
            "company": "Delta",
            "location": "HK",
            "start_yyyy_mm": "2020-01",
            "end_yyyy_mm": "2021-12",
            "bullet_text": "Power BI data modeling at scale",
            "skills": "",
            "tools": "",
            "methods": "",
            "outcome_metric": "",
            "evidence_link": "",
            "confidentiality_ok": "TRUE",
            "recency_score": "0.5",
            "impact_score": "0.6",
            "leadership_score": "0.5",
            "angle": "",
            "keywords_explicit": "",
            "seniority_level": "",
            "function": "",
            "industry": "",
            "domain": "",
            "bullet_group": "",
        },
    ]
    bank_path = write_bank(tmp_path, rows)

    # Current selected bullets already include Role A weaker bullet
    selected = ["Owned dashboards automation for execs"]

    # Alignment: "power bi" is missing
    alignment_report = {
        "targets": {"density_target_atomic": 2},
        "atomic": {
            "term_results": [
                {"term": "power bi", "density": 0, "in_experience": True},
            ]
        },
    }

    jd_atomic = ["power bi", "dashboard"]
    jd_synonyms = {}
    VAR = build_canonical_variants([], jd_atomic, jd_synonyms)

    promos = propose_jd_promotions(
        alignment_report=alignment_report,
        selected_bullets=selected,
        jd_atomic=jd_atomic,
        atomic_synonyms=VAR["atomic_map"],
        bank_path=bank_path,
        max_promotions=3,
        variant_index=VAR["atomic_set"],
    )

    assert promos, "Expected a promotion suggestion"
    # The 'to_bullet' should come from the same role as the selected bullet (Gamma)
    same_role_to = any(
        (p.get("role_tuple") or [None, None])[1] == "Gamma"  # role_tuple = (title, company, ...)
        for p in promos
    )
    assert same_role_to, "Promotion should prefer candidate from same role"