"""
Protein değişikliği ayrıştırıcıları — tüm pipeline scriptlerinde paylaşılan tek kaynak.

Önceden 04_annotate_features.py, 04c_add_context_windows.py ve 04d_add_layer1_scores.py
içinde farklı isimlerle (parse_protein_change / parse_aa_position / parse_aa_change)
tekrar eden bu mantık tek bir fonksiyona indirgendi.
THREE_TO_ONE sözlüğü de yalnızca burada tanımlanmaktadır.
"""

import re

# ── Sabitler ──────────────────────────────────────────────────────────────────

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Ter": "*", "Pyl": "O", "Xaa": "X",
}

_AA_VALID = set(THREE_TO_ONE.values()) - {"*"}

_RE_3L = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
_RE_1L = re.compile(r"p\.([A-Z])(\d+)([A-Z])(?![a-z])")


# ── Ayrıştırıcı ───────────────────────────────────────────────────────────────

def parse_protein_change(pc: str) -> "tuple[str, str, int]":
    """
    ClinVar / VEP / gnomAD protein değişikliği stringini ayrıştır.
    Döndürür: (ref_aa_1L, alt_aa_1L, konum).
    Başarısızlıkta: ('X', 'X', -1).

    Desteklenen formatlar:
      - Üç harfli HGVS:  p.Asp266Val
      - Tek harfli HGVS: p.D266V
      - Tam Name stringi içinde:
          NM_000527.5(LDLR):c.797A>T (p.Asp266Val)
    """
    if not isinstance(pc, str) or not pc:
        return "X", "X", -1

    m = _RE_3L.search(pc)
    if m:
        ref = THREE_TO_ONE.get(m.group(1), "X")
        alt = THREE_TO_ONE.get(m.group(3), "X")
        if ref != "X" and alt != "X":
            return ref, alt, int(m.group(2))

    m = _RE_1L.search(pc)
    if m:
        ref, alt = m.group(1), m.group(3)
        if ref in _AA_VALID and alt in _AA_VALID:
            return ref, alt, int(m.group(2))

    return "X", "X", -1
