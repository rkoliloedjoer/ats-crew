import re
from typing import List, Dict

def _tokenize(text: str) -> str:
    """Lowercase and collapse whitespace so matching is simpler."""
    return re.sub(r"\s+", " ", text.lower()).strip()

def compute_metrics(
    resume_text: str,
    must_terms: List[str],
    synonyms: Dict[str, List[str]],
    density_target: int = 2,
) -> Dict:
    """
    Compute simple ATS-style metrics:

    - coverage: share of must-have terms found at least once (including synonyms)
    - low_density_terms: any must-have term that appears but fewer than `density_target` times
    - placement_ratio: share of must-have terms that appear inside the 'EXPERIENCE' section
    - term_results: per-term details (variants used, counts, whether seen in experience)

    Notes:
    - We look for an 'EXPERIENCE' section, then treat everything until the next ALL-CAPS header
      (or end of text) as experience content.
    - Matching is case-insensitive and uses word boundaries to avoid partial hits.
    """
    text_full = _tokenize(resume_text)

    # Try to isolate the EXPERIENCE block (so we can check placement there)
    exp_block_norm = ""
    # Find 'EXPERIENCE' until the next ALL-CAPS header (e.g., SKILLS, EDUCATION) or end of text
    m = re.search(r"(?is)experience\s*(.+?)(?:\n[A-Z ]{3,}:|$)", resume_text)
    if not m:
        # fallback: everything after the first 'EXPERIENCE'
        m = re.search(r"(?is)experience\s*(.+)$", resume_text)
    if m:
        exp_block_norm = _tokenize(m.group(1))

    results = []
    covered_count = 0
    in_experience_count = 0

    for term in must_terms:
        term = (term or "").strip()
        if not term:
            continue

        # Build variant list from synonyms map (canonical term -> variants[])
        variants = [term.lower()]
        for canon, vs in (synonyms or {}).items():
            if str(canon).lower() == term.lower():
                variants.extend([str(v).lower() for v in (vs or [])])

        # Deduplicate variants while preserving order
        seen = set()
        vlist = []
        for v in variants:
            if v not in seen:
                vlist.append(v); seen.add(v)

        # Count appearances in the full resume (density)
        density = 0
        for v in vlist:
            # word-boundary regex for safe matching
            re_word = re.compile(rf"\b{re.escape(v)}\b", re.I)
            density += len(re_word.findall(text_full))

        # Check if any variant appears inside the EXPERIENCE block
        in_experience = False
        if exp_block_norm:
            for v in vlist:
                if re.search(rf"\b{re.escape(v)}\b", exp_block_norm, flags=re.I):
                    in_experience = True
                    break

        if density > 0:
            covered_count += 1
        if in_experience:
            in_experience_count += 1

        results.append({
            "term": term,
            "variants": vlist,
            "density": density,
            "in_experience": in_experience
        })

    coverage = (covered_count / len(must_terms)) if must_terms else 0.0
    placement_ratio = (in_experience_count / len(must_terms)) if must_terms else 0.0

    low_density_terms = [
        {"term": r["term"], "density": r["density"]}
        for r in results
        if 0 < r["density"] < density_target
    ]
    missing_terms = [r["term"] for r in results if r["density"] == 0]
    placement_issues = [r["term"] for r in results if not r["in_experience"]]

    return {
        "coverage": coverage,
        "missing_terms": missing_terms,
        "low_density_terms": low_density_terms,
        "placement_ratio": placement_ratio,
        "term_results": results
    }