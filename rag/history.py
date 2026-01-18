import os
import time
import boto3
from boto3.dynamodb.conditions import Key

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


class DynamoDBChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain-style chat history backed by DynamoDB.
    One item per message.
    """

    def __init__(
        self,
        session_id: str,
        table_name: str | None = None,
        region: str | None = None,
        user_id: str = "",
        limit: int = 20,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.limit = limit

        self.region = (
            region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.table_name = table_name or os.getenv(
            "DDB_TABLE", "rag_chat_history"
        )

        self._table = boto3.resource(
            "dynamodb", region_name=self.region
        ).Table(self.table_name)

    # -------- LangChain required API --------

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Load the last N messages for this session, ordered oldest → newest.
        """
        resp = self._table.query(
            KeyConditionExpression=Key("session_id").eq(self.session_id),
            ScanIndexForward=False,  # newest first
            Limit=self.limit,
        )

        items = list(reversed(resp.get("Items", [])))

        out: list[BaseMessage] = []
        for i in items:
            role = i.get("role")
            text = i.get("message", "")

            if role == "user":
                out.append(HumanMessage(content=text))
            elif role == "assistant":
                out.append(AIMessage(content=text))
            elif role == "system":
                out.append(SystemMessage(content=text))
            else:
                # fallback
                out.append(HumanMessage(content=text))

        return out

    def add_message(self, message: BaseMessage) -> None:
        """
        Required by BaseChatMessageHistory.
        """
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"

        self._put_item(role=role, text=message.content)

    def clear(self) -> None:
        """
        Optional; not implemented to avoid table scans.
        Use TTL instead if you need cleanup.
        """
        raise NotImplementedError("Use TTL or batch delete externally")

    # -------- Convenience helpers --------

    def add_user_message(self, text: str) -> None:
        self._put_item(role="user", text=text, extra=None)

    def add_ai_message(self, text: str, trace_id: str | None = None) -> None:
        extra: dict[str, str] = {}
        if trace_id:
            extra["trace_id"] = trace_id
        self._put_item(role="assistant", text=text, extra=extra or None)

    # -------- Internal --------

    def _put_item(self, role: str, text: str, extra: dict | None = None) -> None:
        item = {
            "session_id": self.session_id,
            "ts": int(time.time() * 1000),
            "role": role,
            "user_id": self.user_id,
            "message": text,
        }
        if extra:
            item.update(extra)

        self._table.put_item(Item=item)