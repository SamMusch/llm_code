# app.py
import streamlit as st
from rag.i_pipe import get_db
from rag.g_pipe import rag_function

HF_API_KEY = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


@st.cache_resource(show_spinner=False)
def _db():
    return get_db()

st.title("Mini RAG")
q = st.text_input("Ask a question")
k = st.slider("k (retrieved chunks)", 1, 8, 3, 1)

if st.button("Run") and q.strip():
    ctx, ans = rag_function(q, k=k, db=_db())
    st.subheader("Answer")
    st.write(ans)
    with st.expander("Retrieved context"):
        for i, c in enumerate(ctx, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(c)
