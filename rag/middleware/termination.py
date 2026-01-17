# rag/middleware/termination.py
from langchain.agents.middleware import AgentState, before_model

@before_model
def stop_after_final_answer(state: AgentState, runtime):
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]
    if getattr(last, "role", None) == "assistant":
        # Final answer already produced → stop graph
        runtime.stop()
    return None