# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
os.environ.setdefault("USER_AGENT", "rag-app/0.1 (+local)")

# load .env once for the whole package
ROOT = Path(__file__).resolve().parents[1]
# load_dotenv(ROOT / ".env")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")


# shared constants
# HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def get_hf_client():
    # lightweight factory; avoids side effects on import
    from huggingface_hub import InferenceClient
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model=HF_MODEL, token=token)
