"""
gnomAD API yardımcı fonksiyonları — tüm pipeline scriptlerinde paylaşılan tek kaynak.

Önceden 02_fetch_gnomad.py, 03b_fix_single_gene_panels.py ve 03c_fix_pah_panel.py
içinde tekrar eden aşağıdaki öğeler buraya taşındı:
  - GNOMAD_API, POPULATION_IDS, QUERY, MAX_RETRIES, SLEEP_SEC sabitleri
  - query_gene()
  - get_af()
  - get_pop_af()
  - clean_hgvsp()
"""

import time
import logging
import requests

# ── Sabitler ──────────────────────────────────────────────────────────────────

GNOMAD_API     = "https://gnomad.broadinstitute.org/api"
POPULATION_IDS = ["afr", "amr", "eas", "fin", "nfe", "sas"]
MAX_RETRIES    = 4
SLEEP_SEC      = 2.0

QUERY = """
query($gene: String!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    variants(dataset: gnomad_r4) {
      variant_id
      pos
      ref
      alt
      hgvsp
      consequence
      genome { af ac an
        populations { id ac an }
      }
      exome  { af ac an
        populations { id ac an }
      }
    }
  }
}
"""

log = logging.getLogger(__name__)


# ── API yardımcıları ──────────────────────────────────────────────────────────

def query_gene(gene: str) -> "list | None":
    """gnomAD API'den bir genin tüm varyantlarını sorgula.

    Dönüş değerleri:
      list  → API'den alınmış geçerli cevap (boş liste dahil — gen bulunamadı
              veya API deterministik hata döndürdü; yeniden denemek sonucu
              değiştirmez, checkpoint'lenebilir)
      None  → tüm denemeler ağ/timeout hatasıyla tükendi (GEÇİCİ hata —
              checkpoint YAZILMAMALI, sonraki çalıştırmada yeniden denenmeli)
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                GNOMAD_API,
                json={"query": QUERY, "variables": {"gene": gene}},
                timeout=180,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                log.warning(f"  API errors for {gene}: {data['errors']}")
                return []
            gene_data = (data.get("data") or {}).get("gene")
            if not gene_data:
                log.warning(f"  No gene data returned for: {gene}")
                return []
            return gene_data.get("variants") or []
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout ({attempt}/{MAX_RETRIES}) for {gene}")
            time.sleep(5 * attempt)
        except Exception as e:
            log.warning(f"  Error ({attempt}/{MAX_RETRIES}) for {gene}: {e}")
            time.sleep(5 * attempt)
    return None


def get_af(variant: dict, source: str) -> float:
    """Genome veya exome kaynağından allel frekansını al."""
    src = variant.get(source) or {}
    return float(src.get("af") or 0.0)


def get_pop_af(variant: dict, pop_id: str) -> float:
    """Belirli bir popülasyon için allel frekansını çek (genome önce, exome fallback)."""
    for source in ["genome", "exome"]:
        src  = variant.get(source) or {}
        pops = src.get("populations") or []
        for p in pops:
            if p.get("id", "").lower() == pop_id:
                an = p.get("an") or 0
                if an > 0:
                    return (p.get("ac") or 0) / an
    return float("nan")


def clean_hgvsp(hgvsp: str) -> str:
    """
    gnomAD'ın ENSP önekli hgvsp stringlerini normalize et.
    'ENSP00000269305.4:p.Asp266Val' → 'p.Asp266Val'
    """
    if not isinstance(hgvsp, str):
        return ""
    if ":" in hgvsp:
        hgvsp = hgvsp.split(":")[-1]
    return hgvsp.strip()
