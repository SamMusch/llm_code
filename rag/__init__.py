# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
os.environ.setdefault("USER_AGENT", "rag-app/0.1 (+local)")

ROOT = Path(__file__).resolve().parents[1]       # load .env once for the whole package
load_dotenv(ROOT / ".env")
