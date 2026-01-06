# rag/history.py
import os, time, boto3
from boto3.dynamodb.conditions import Key
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

class DynamoDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, table_name: str | None = None, region: str | None = None, user_id: str = ""):
        self.session_id = session_id
        self.user_id = user_id
        self.region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.table_name = table_name or os.getenv("DDB_TABLE", "rag_chat_history")
        self.table = boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    @property
    def messages(self) -> list[BaseMessage]:
        resp = self.table.query(
            KeyConditionExpression=Key("session_id").eq(self.session_id),
            ScanIndexForward=True,
        )
        out: list[BaseMessage] = []
        for i in resp.get("Items", []):
            role, text = i["role"], i["message"]
            out.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
        return out

    def add_message(self, message: BaseMessage) -> None:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        self.table.put_item(Item={
            "session_id": self.session_id,
            "ts": int(time.time() * 1000),
            "role": role,
            "user_id": self.user_id,
            "message": message.content,
        })

    def clear(self) -> None:
        # optional: delete items (needs Scan + BatchWrite or TTL)
        raise NotImplementedError