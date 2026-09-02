"""Shared logging setup — tek yerden yapılandır, her scriptte import et."""

import os
import sys
import logging


def get_logger(log_filename: str) -> logging.Logger:
    """
    logs/ dizinine dosya + stdout handler'lı logger döndürür.
    Birden fazla çağrı aynı handler'ı iki kez eklemez.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_filename)
    logger   = logging.getLogger(log_filename)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(message)s")

        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger
