# -*- coding: utf-8 -*-

from rag.i_pipe import get_db               # I
from rag.g_pipe import rag_function         # RAG

def run(query: str, k: int = 3):
    db = get_db()
    ctx, ans = rag_function(query, k=k, db=db)
    return ctx, ans

if __name__ == "__main__":
    _, ans = run("Who won?", k=2)
    print(ans)
