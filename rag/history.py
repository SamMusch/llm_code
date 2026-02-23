import os
from typing import Optional

from langchain_community.chat_message_histories.dynamodb import DynamoDBChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


def _norm(s: Optional[str], default: str) -> str:
    v = (s or default).strip().lower()
    return v or default


# Local-only in-memory history store (keyed by fully-scoped session id).
_LOCAL_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}


def _local_history(scoped_session_id: str) -> InMemoryChatMessageHistory:
    h = _LOCAL_HISTORIES.get(scoped_session_id)
    if h is None:
        h = InMemoryChatMessageHistory()
        _LOCAL_HISTORIES[scoped_session_id] = h
    return h


def build_chat_history(
    session_id: str,
    *,
    # Caller should pass the authenticated principal id in AWS (e.g., Cognito sub).
    # For local usage, this can be left None and will fall back to USER_ID or "local".
    principal_id: Optional[str] = None,
    env: Optional[str] = None,
    table_name: Optional[str] = None,
    region_name: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    history_size: Optional[int] = 20,
) -> BaseChatMessageHistory:
    """Create a LangChain DynamoDBChatMessageHistory with strong env/user scoping.

    We isolate records by namespacing the session_id:
        <env>#<principal_id>#<session_id>

    This prevents local (ollama) and aws (ecs/bedrock) sessions from colliding.

    LangChain reference:
    https://reference.langchain.com/v0.3/python/community/chat_message_histories/langchain_community.chat_message_histories.dynamodb.DynamoDBChatMessageHistory.html
    """

    app_env = _norm(env or os.getenv("APP_ENV"), "local")
    pid = (principal_id or os.getenv("USER_ID") or "local").strip()

    # Critical: make collisions impossible across environments and users.
    scoped_session_id = f"{app_env}#{pid}#{session_id}".strip()

    # Local mode: never touch DynamoDB. Keep chat history in memory.
    if app_env == "local":
        return _local_history(scoped_session_id)

    tname = table_name or os.getenv("DDB_TABLE", "rag_chat_history")
    region = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be set")

    # Note: DynamoDBChatMessageHistory stores the conversation as a single item
    # under History (a list of serialized messages).
    return DynamoDBChatMessageHistory(
        table_name=tname,
        session_id=scoped_session_id,
        primary_key_name="SessionId",
        history_messages_key="History",
        ttl=ttl_seconds,
        ttl_key_name=os.getenv("DDB_TTL_KEY", "expireAt"),
        history_size=history_size,
        boto3_session=None,  # uses default boto3 resolution chain
    )