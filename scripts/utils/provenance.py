"""
Veri provenance (kaynak izleme) yardımcıları.

Her fetch adımı, çıktısının yanına bir provenance JSON dosyası yazar:
hangi kaynaktan, hangi tarihte, hangi sürümden, hangi filtrelerle çekildi.
ClinVar variant_summary haftalık güncellendiği için bu kayıt olmadan
veri seti tekrarlanabilir değildir.

Kullanım:
    from utils.provenance import write_provenance, file_md5

    write_provenance(
        "data/raw/provenance_clinvar.json",
        source="ClinVar variant_summary",
        url=URL,
        source_version=last_modified_header,   # ClinVar sürüm göstergesi
        input_files={"variant_summary.txt.gz": file_md5(GZ_PATH)},
        row_count=len(df),
        filters={"assembly": "GRCh38", "review": sorted(HIGH_CONF)},
    )
"""

import os
import json
import hashlib
from datetime import datetime, timezone


def file_md5(path: str, chunk_size: int = 1 << 20) -> str | None:
    """Dosyanın MD5 özetini hesapla. Dosya yoksa None döner."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(out_path: str,
                     source: str,
                     url: str | None = None,
                     source_version: str | None = None,
                     input_files: dict | None = None,
                     row_count: int | None = None,
                     filters: dict | None = None,
                     extra: dict | None = None) -> dict:
    """
    Provenance kaydını JSON olarak yaz ve dict olarak döndür.

    Parameters
    ----------
    out_path : str
        Yazılacak JSON dosya yolu (örn. data/raw/provenance_clinvar.json).
    source : str
        Veri kaynağı adı (örn. "ClinVar variant_summary", "gnomAD GraphQL r4").
    url : str | None
        Kaynak URL/endpoint.
    source_version : str | None
        Kaynak sürüm göstergesi (HTTP Last-Modified, dataset adı, release ID...).
    input_files : dict | None
        {dosya_adı: md5} eşlemesi.
    row_count : int | None
        Çıktıdaki satır sayısı.
    filters : dict | None
        Uygulanan filtreler (eşikler, review status seti...).
    extra : dict | None
        Kaynağa özgü ek alanlar (başarısız genler, checkpoint sayısı...).
    """
    record = {
        "source": source,
        "url": url,
        "source_version": source_version,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": input_files or {},
        "row_count": row_count,
        "filters": filters or {},
    }
    if extra:
        record.update(extra)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record
