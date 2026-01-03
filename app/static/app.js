const chatLog = document.getElementById("chatLog");
const chatMsg = document.getElementById("chatMsg");
const sendBtn = document.getElementById("sendBtn");

function appendLine(role, text) {
  const div = document.createElement("div");
  div.style.marginBottom = "10px";
  div.innerHTML = `<strong>${role}:</strong> <span></span>`;
  div.querySelector("span").textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div.querySelector("span");
}

function streamAnswer(message) {
  const user = message.trim();
  if (!user) return;

  appendLine("You", user);
  const outSpan = appendLine("Bot", "");

  const url = `/chat/stream?message=${encodeURIComponent(user)}`;
  const es = new EventSource(url);

  es.addEventListener("start", () => {});
  es.addEventListener("error", (e) => {
    es.close();
    outSpan.textContent += "\n[stream error]";
  });
  es.addEventListener("end", () => {
    es.close();
  });

  es.onmessage = (evt) => {
    const chunk = (evt.data || "").replaceAll("\\n", "\n");
    outSpan.textContent += chunk;
    chatLog.scrollTop = chatLog.scrollHeight;
  };
}

sendBtn?.addEventListener("click", () => {
  const msg = chatMsg.value;
  chatMsg.value = "";
  streamAnswer(msg);
});

chatMsg?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const msg = chatMsg.value;
    chatMsg.value = "";
    streamAnswer(msg);
  }
});