"""
Pipeline Test Runner — Pre/Post Optimization Verification
==========================================================
10 test case'i utils modülleri üzerinde çalıştırır.
Sonuçları tests/output/test_results_after.json dosyasına yazar.

Kullanım (repo root'tan):
  python scripts/run_tests.py            # utils üzerinden test (varsayılan)
  python scripts/run_tests.py --baseline # refactor ÖNCESİ kod üzerinde baseline al
                                         # (yalnızca pre-refactor commit'lerde çalışır)

Karşılaştırma:
  python scripts/run_tests.py --compare  # baseline ile farkı göster
"""

import sys
import os
import re
import json
import math
import argparse
import importlib.util
from datetime import datetime

# ── Repo root'a geç ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("logs", exist_ok=True)  # scripts import öncesi oluştur

# ── Kaynak seçici ─────────────────────────────────────────────────────────────

def load_module(filename, mod_name):
    path = os.path.join(ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_functions_from_scripts():
    """Refactor ÖNCESİ scriptlerden fonksiyonları yükle (baseline modu).

    Refactor sonrası 01_fetch_clinvar.py bu fonksiyonları utils'ten import
    ettiği için modül üzerinde artık bulunmazlar. Bu durumda anlaşılır bir
    hata ile çıkılır — baseline yalnızca pre-refactor commit'lerde alınabilir.
    """
    m01 = load_module("01_fetch_clinvar.py", "clinvar01")
    required = ["assign_label", "assign_label_with_confidence",
                "extract_ref_alt", "is_missense"]
    missing = [name for name in required if not hasattr(m01, name)]
    if missing:
        sys.exit(
            "HATA: 01_fetch_clinvar.py içinde şu fonksiyonlar bulunamadı: "
            f"{missing}\n"
            "Bu kod refactor edilmiş — fonksiyonlar scripts/utils/ altına taşındı.\n"
            "Baseline almak için refactor ÖNCESİ bir commit'e geçin, ya da\n"
            "normal test için flag'siz çalıştırın: python scripts/run_tests.py"
        )
    return {name: getattr(m01, name) for name in required}


def get_functions_from_utils():
    """Refactor edilmiş utils modülünden fonksiyonları yükle (post-optimization)."""
    utils_path = os.path.join(ROOT, "scripts", "utils", "clinvar_utils.py")
    if not os.path.exists(utils_path):
        raise FileNotFoundError(
            "scripts/utils/clinvar_utils.py bulunamadı — önce optimizasyonu tamamlayın."
        )
    spec = importlib.util.spec_from_file_location("clinvar_utils", utils_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "assign_label":                mod.assign_label,
        "assign_label_with_confidence": mod.assign_label_with_confidence,
        "extract_ref_alt":             mod.extract_ref_alt,
        "is_missense":                 mod.is_missense,
    }


# ── Serialize yardımcısı ──────────────────────────────────────────────────────

def serialize(val):
    """JSON uyumlu hale getir: None, NaN, tuple → list."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return "NaN"
    if isinstance(val, tuple):
        return [serialize(v) for v in val]
    if isinstance(val, list):
        return [serialize(v) for v in val]
    return val


# ── 10 Test Case ──────────────────────────────────────────────────────────────

def run_all_tests(fns: dict) -> list:
    al  = fns["assign_label"]
    alc = fns["assign_label_with_confidence"]
    era = fns["extract_ref_alt"]
    im  = fns["is_missense"]

    TEST_CASES = [
        # TC-01: Açık pathogenic
        {
            "id": "TC-01",
            "function": "assign_label",
            "input": "Pathogenic",
            "expected": 1,
            "actual": al("Pathogenic"),
        },
        # TC-02: Açık benign
        {
            "id": "TC-02",
            "function": "assign_label",
            "input": "Benign",
            "expected": 0,
            "actual": al("Benign"),
        },
        # TC-03: Conflicting (EXCLUDE_SIG listesinde)
        {
            "id": "TC-03",
            "function": "assign_label",
            "input": "Conflicting interpretations of pathogenicity",
            "expected": None,
            "actual": al("Conflicting interpretations of pathogenicity"),
        },
        # TC-04: Hem pathogenic hem benign → None
        {
            "id": "TC-04",
            "function": "assign_label",
            "input": "Pathogenic/Benign",
            "expected": None,
            "actual": al("Pathogenic/Benign"),
        },
        # TC-05: None input → None (type guard)
        {
            "id": "TC-05",
            "function": "assign_label",
            "input": None,
            "expected": None,
            "actual": al(None),
        },
        # TC-06: Likely pathogenic → (1, "likely")
        {
            "id": "TC-06",
            "function": "assign_label_with_confidence",
            "input": "Likely pathogenic",
            "expected": [1, "likely"],
            "actual": alc("Likely pathogenic"),
        },
        # TC-07: Hard pathogenic → (1, "hard")
        {
            "id": "TC-07",
            "function": "assign_label_with_confidence",
            "input": "Pathogenic",
            "expected": [1, "hard"],
            "actual": alc("Pathogenic"),
        },
        # TC-08: HGVS c.35G>T → ("G","T")
        {
            "id": "TC-08",
            "function": "extract_ref_alt",
            "input": "NM_004985.5(KRAS):c.35G>T (p.Gly12Val)",
            "expected": ["G", "T"],
            "actual": era("NM_004985.5(KRAS):c.35G>T (p.Gly12Val)"),
        },
        # TC-09: Geçerli missense → True
        {
            "id": "TC-09",
            "function": "is_missense",
            "input": "NM_004985.5(KRAS):c.35G>T (p.Gly12Val)",
            "expected": True,
            "actual": im("NM_004985.5(KRAS):c.35G>T (p.Gly12Val)"),
        },
        # TC-10: Stop kazanımı (Ter) → False
        {
            "id": "TC-10",
            "function": "is_missense",
            "input": "NM_000527.5(LDLR):c.907T>A (p.Cys303Ter)",
            "expected": False,
            "actual": im("NM_000527.5(LDLR):c.907T>A (p.Cys303Ter)"),
        },
    ]

    results = []
    for tc in TEST_CASES:
        expected_s = serialize(tc["expected"])
        actual_s   = serialize(tc["actual"])
        passed     = (expected_s == actual_s)
        results.append({
            "id":       tc["id"],
            "function": tc["function"],
            "input":    serialize(tc["input"]),
            "expected": expected_s,
            "actual":   actual_s,
            "passed":   passed,
        })
    return results


# ── Karşılaştırma ─────────────────────────────────────────────────────────────

def compare_with_baseline(results: list, baseline_path: str) -> None:
    if not os.path.exists(baseline_path):
        print(f"Baseline bulunamadı: {baseline_path}")
        return
    with open(baseline_path) as f:
        baseline = json.load(f)

    base_map = {r["id"]: r for r in baseline["results"]}
    print("\n── Karşılaştırma (Baseline vs Yeni) ──────────────────────────────")
    all_match = True
    for r in results:
        tc_id = r["id"]
        b = base_map.get(tc_id)
        if b is None:
            print(f"  {tc_id}: baseline'da yok")
            continue
        if r["actual"] == b["actual"]:
            print(f"  {tc_id}: ✓ output aynı")
        else:
            all_match = False
            print(f"  {tc_id}: ✗ FARK — baseline={b['actual']}  yeni={r['actual']}")
    if all_match:
        print("\n✓ Tüm outputlar baseline ile aynı — optimizasyon başarılı!")
    else:
        print("\n✗ Bazı outputlar değişti — kod davranışı değişmiş, kontrol edin!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline test runner")
    parser.add_argument("--baseline", action="store_true",
                        help="Refactor öncesi scriptlerden baseline al "
                             "(yalnızca pre-refactor commit'lerde çalışır)")
    parser.add_argument("--from-utils", action="store_true",
                        help="(varsayılan davranış; geriye dönük uyumluluk için korundu)")
    parser.add_argument("--compare", action="store_true",
                        help="Sonuçları baseline ile karşılaştır")
    args = parser.parse_args()

    use_baseline = args.baseline and not args.from_utils
    source_label = ("scripts (pre-optimization)" if use_baseline
                    else "utils (post-optimization)")
    print(f"\n=== Pipeline Test Runner ===")
    print(f"Kaynak: {source_label}")
    print(f"Zaman:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if use_baseline:
        fns = get_functions_from_scripts()
    else:
        fns = get_functions_from_utils()

    results = run_all_tests(fns)

    # Özet
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"{'ID':<8} {'Fonksiyon':<32} {'Durum'}")
    print("─" * 55)
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {r['id']:<6} {r['function']:<32} {status}")
        if not r["passed"]:
            print(f"         expected={r['expected']}  actual={r['actual']}")
    print("─" * 55)
    print(f"Geçen: {passed}/10   Kalan: {failed}/10\n")

    # Kaydet
    output = {
        "metadata": {
            "run_date":    datetime.now().isoformat(),
            "source":      source_label,
            "total_tests": len(results),
            "passed":      passed,
            "failed":      failed,
        },
        "results": results,
    }

    if use_baseline:
        out_path = os.path.join(ROOT, "tests", "baseline", "test_results.json")
    else:
        out_path = os.path.join(ROOT, "tests", "output", "test_results_after.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Sonuçlar kaydedildi → {out_path}")

    # Karşılaştırma istenirse
    if args.compare:
        baseline_path = os.path.join(ROOT, "tests", "baseline", "test_results.json")
        compare_with_baseline(results, baseline_path)

    # CI/orkestrasyon için anlamlı çıkış kodu
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
