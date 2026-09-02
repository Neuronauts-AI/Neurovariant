"""
Pipeline Output Karşılaştırma Scripti
======================================
Optimizasyon öncesi (baseline) ve sonrası test sonuçlarını karşılaştırır.
Her test case'in input ve output değerlerinin birebir aynı olduğunu doğrular.

Kullanım (repo root'tan):
  python3 scripts/compare_results.py
  python3 scripts/compare_results.py --before path/to/before.json --after path/to/after.json

Çıkış kodu:
  0 → tüm karşılaştırmalar eşleşti (optimizasyon güvenli)
  1 → en az bir fark bulundu (davranış değişti — incele!)
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BEFORE = os.path.join(ROOT, "tests", "baseline", "test_results.json")
DEFAULT_AFTER  = os.path.join(ROOT, "tests", "output", "test_results_after.json")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"{RED}HATA: Dosya bulunamadı → {path}{RESET}")
        print("  Önce şunu çalıştırın:")
        print("    python3 scripts/run_tests.py             # utils üzerinden test çalıştır")
        print("    (baseline tests/baseline/ altında commit'li olarak gelir)")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(before: dict, after: dict) -> bool:
    """
    İki sonuç dosyasındaki her test case'i karşılaştırır.
    Her case için input ve output ayrı ayrı kontrol edilir.
    True → tüm eşleşti, False → en az bir fark var.
    """
    before_map = {r["id"]: r for r in before["results"]}
    after_map  = {r["id"]: r for r in after["results"]}

    all_ids = sorted(set(before_map) | set(after_map))
    all_ok  = True

    print(f"\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}{'ID':<8} {'Fonksiyon':<33} {'Input':<6} {'Output':<8} Durum{RESET}")
    print(f"{BOLD}{'─'*65}{RESET}")

    for tc_id in all_ids:
        b = before_map.get(tc_id)
        a = after_map.get(tc_id)

        if b is None:
            print(f"  {tc_id:<6} {'?':<33} {YELLOW}Baseline'da yok{RESET}")
            all_ok = False
            continue
        if a is None:
            print(f"  {tc_id:<6} {'?':<33} {YELLOW}After dosyasında yok{RESET}")
            all_ok = False
            continue

        input_match  = (b["input"]    == a["input"])
        output_match = (b["actual"]   == a["actual"])
        expected_match = (b["expected"] == a["expected"])
        row_ok = input_match and output_match and expected_match

        if row_ok:
            status = f"{GREEN}✓ AYNI{RESET}"
        else:
            status = f"{RED}✗ FARK{RESET}"
            all_ok = False

        in_mark  = f"{GREEN}✓{RESET}" if input_match    else f"{RED}✗{RESET}"
        out_mark = f"{GREEN}✓{RESET}" if output_match   else f"{RED}✗{RESET}"
        fn_name  = b.get("function", "?")
        print(f"  {tc_id:<6} {fn_name:<33} {in_mark}      {out_mark}       {status}")

        # Fark varsa detay göster
        if not input_match:
            print(f"         {YELLOW}input değişti!{RESET}")
            print(f"           önceki : {b['input']}")
            print(f"           sonraki: {a['input']}")
        if not output_match:
            print(f"         {YELLOW}output değişti!{RESET}")
            print(f"           önceki : {b['actual']}")
            print(f"           sonraki: {a['actual']}")
        if not expected_match:
            print(f"         {YELLOW}expected değişti!{RESET}")
            print(f"           önceki : {b['expected']}")
            print(f"           sonraki: {a['expected']}")

    print(f"{BOLD}{'─'*65}{RESET}")

    # Metadata karşılaştırması
    b_meta = before.get("metadata", {})
    a_meta = after.get("metadata",  {})
    print(f"\n{BOLD}Metadata:{RESET}")
    print(f"  Baseline    — tarih: {b_meta.get('run_date','?')}  "
          f"geçen: {b_meta.get('passed','?')}/{b_meta.get('total_tests','?')}")
    print(f"  Optimizasyon— tarih: {a_meta.get('run_date','?')}  "
          f"geçen: {a_meta.get('passed','?')}/{a_meta.get('total_tests','?')}")

    # Sonuç
    total  = len(all_ids)
    passed = sum(
        1 for tc_id in all_ids
        if before_map.get(tc_id) and after_map.get(tc_id)
        and before_map[tc_id]["input"]   == after_map[tc_id]["input"]
        and before_map[tc_id]["actual"]  == after_map[tc_id]["actual"]
        and before_map[tc_id]["expected"] == after_map[tc_id]["expected"]
    )

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}✓ BAŞARILI — {passed}/{total} test case "
              f"birebir aynı. Optimizasyon davranışı değiştirmedi.{RESET}")
    else:
        diff = total - passed
        print(f"{RED}{BOLD}✗ BAŞARISIZ — {diff}/{total} test case farklı. "
              f"Kod davranışı değişti, kontrol edin!{RESET}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Pipeline output karşılaştırma aracı")
    parser.add_argument("--before", default=DEFAULT_BEFORE,
                        help="Optimizasyon öncesi JSON (varsayılan: tests/baseline/test_results.json)")
    parser.add_argument("--after",  default=DEFAULT_AFTER,
                        help="Optimizasyon sonrası JSON (varsayılan: tests/output/test_results_after.json)")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Pipeline Output Karşılaştırma ==={RESET}")
    print(f"  Önceki : {args.before}")
    print(f"  Sonraki: {args.after}")

    before = load(args.before)
    after  = load(args.after)

    ok = compare(before, after)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
