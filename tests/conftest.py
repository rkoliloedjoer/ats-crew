# tests/conftest.py
import sys, pathlib
from pathlib import Path
import pandas as pd

# Ensure src/ and project root are importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.append(p)

# Minimal bank writer you can reuse
def mk_bank(tmp_path: Path, rows: list[dict]) -> str:
    bank_path = tmp_path / "src" / "bank" / "experience_bank.csv"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    required = [
        "role_title","company","location","start_yyyy_mm","end_yyyy_mm",
        "seniority_level","function","industry","domain","bullet_text",
        "skills","tools","methods","outcome_metric","evidence_link",
        "confidentiality_ok","recency_score","impact_score","leadership_score",
        "angle","keywords_explicit"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = ""
    df.to_csv(bank_path, index=False)
    return str(bank_path)
