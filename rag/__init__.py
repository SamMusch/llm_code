# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
import os
import logging
import random
import numpy as np
import streamlit as st
os.environ.setdefault("USER_AGENT", "rag-app/0.1 (+local)")

ROOT = Path(__file__).resolve().parents[1]       # load .env once for the whole package
load_dotenv(ROOT / ".env")



# ----------------------------------------
# OBSERVABILIITY bootstrap
# Extract metadata from each step
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_JSON = os.environ.get("LOG_JSON", "false").lower() == "true"
if LOG_JSON:
    log_format = '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
else:
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=log_format)
logger = logging.getLogger("rag")
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
logger.info(
    "Initialized observability",
    extra={
        "phase": "init",
        "node": "__init__",
        "num_docs": 0,
        "latency_ms": 0,
        "seed": DEFAULT_SEED,
    },
)
