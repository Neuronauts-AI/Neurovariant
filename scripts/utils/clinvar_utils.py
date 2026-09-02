"""
ClinVar yardımcı fonksiyonları — tüm pipeline scriptlerinde paylaşılan tek kaynak.

Önceden 01_fetch_clinvar.py, 01b_supplement_benign.py ve 03_build_panels.py
içinde kelimesi kelimesine kopyalanmış olan aşağıdaki öğeler buraya taşındı:
  - EXCLUDE_SIG, HGVS_SNV_RE, MISSENSE_RE, NON_MISSENSE_RE sabitleri
  - assign_label()
  - assign_label_with_confidence()
  - extract_ref_alt()
  - is_missense()
"""

import re

# ── Sabitler ──────────────────────────────────────────────────────────────────

EXCLUDE_SIG = [
    "conflicting", "uncertain", "not provided", "drug response",
    "association", "protective", "other", "risk factor",
]

HGVS_SNV_RE = re.compile(r"c\.\d+([ACGT])>([ACGT])", re.IGNORECASE)
MISSENSE_RE  = re.compile(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}")
NON_MISSENSE_RE = re.compile(
    r"p\..*(?:Ter|ter|\*|fs|del|ins|dup|Ext|ext|Met1\?|=)", re.IGNORECASE
)


# ── Etiket yardımcıları ────────────────────────────────────────────────────────

def assign_label(sig: str) -> "int | None":
    if not isinstance(sig, str):
        return None
    s = sig.lower().strip()
    if any(kw in s for kw in EXCLUDE_SIG):
        return None
    has_patho  = "pathogenic" in s
    has_benign = "benign" in s
    if has_patho and has_benign:
        return None
    if has_patho:
        return 1
    if has_benign:
        return 0
    return None


def assign_label_with_confidence(sig: str) -> "tuple[int | None, str | None]":
    """
    (label, confidence) döndürür. confidence:
      'hard'   — Pathogenic / Benign (Likely niteleyicisi yok)
      'likely' — Likely Pathogenic / Likely Benign
      None     — dışlanan veya çelişkili
    """
    if not isinstance(sig, str):
        return None, None
    s = sig.lower().strip()
    if any(kw in s for kw in EXCLUDE_SIG):
        return None, None
    has_patho  = "pathogenic" in s
    has_benign = "benign" in s
    if has_patho and has_benign:
        return None, None
    if has_patho:
        return 1, ("likely" if "likely" in s else "hard")
    if has_benign:
        return 0, ("likely" if "likely" in s else "hard")
    return None, None


# ── HGVS ayrıştırıcıları ──────────────────────────────────────────────────────

def extract_ref_alt(name: str) -> "tuple[str, str]":
    """HGVS c. notasyonundan ref ve alt nükleotidleri çıkar."""
    if not isinstance(name, str):
        return "N", "N"
    m = HGVS_SNV_RE.search(name)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    return "N", "N"


def is_missense(name: str) -> bool:
    """Name sütununda missense protein değişikliği varsa True döndür."""
    if not isinstance(name, str):
        return False
    if "p." not in name:
        return False
    if NON_MISSENSE_RE.search(name):
        return False
    return bool(MISSENSE_RE.search(name))
