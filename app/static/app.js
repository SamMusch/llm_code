const el = (id) => document.getElementById(id);

const chatListEl = el("chatList");
const chatTitleEl = el("chatTitle");
const messagesEl = el("chatMessages");
const inputEl = el("chatInput");
const sendBtn = el("sendBtn");
const newChatBtn = el("newChatBtn");

// In-memory UI state (sessions + active session)
let sessions = []; // [{session_id,last_ts,title}]
let activeSessionId = "";
let activeMessages = []; // [{role, content, ts?}]

function nowIso() {
  return new Date().toISOString();
}

function esc(s) {
  return (s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function newSessionId() {
  // simple client-generated id; server treats it as an opaque string
  // (we'll migrate to server-issued ids later if desired)
  return "s_" + Date.now() + "_" + Math.random().toString(16).slice(2);
}

function bubble(role, content) {
  const wrap = document.createElement("div");
  wrap.className = "cg-msg " + (role === "user" ? "user" : "assistant");

  const inner = document.createElement("div");
  inner.className = "cg-bubble";
  inner.innerHTML = esc(content).replaceAll("\n", "<br/>");

  wrap.appendChild(inner);
  return { wrap, inner };
}

function renderMessages(msgs) {
  messagesEl.innerHTML = "";
  (msgs || []).forEach((m) => {
    const { wrap } = bubble(m.role, m.content);
    messagesEl.appendChild(wrap);
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderChatList(list) {
  chatListEl.innerHTML = "";

  (list || [])
    .slice()
    .sort((a, b) => (b.last_ts || 0) - (a.last_ts || 0))
    .forEach((s) => {
      const btn = document.createElement("button");
      btn.className =
        "cg-chatitem" + (s.session_id === activeSessionId ? " active" : "");
      btn.type = "button";
      btn.textContent = (s.title || "Untitled").trim() || "Untitled";
      btn.addEventListener("click", () => {
        openSession(s.session_id);
      });
      chatListEl.appendChild(btn);
    });
}

function autosizeTextarea() {
  if (!inputEl) return;
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

function setHeaderTitle() {
  if (!chatTitleEl) return;
  const found = sessions.find((s) => s.session_id === activeSessionId);
  chatTitleEl.textContent = (found?.title || "New chat").trim() || "New chat";
}

async function fetchJson(url) {
  const r = await fetch(url, { credentials: "same-origin" });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText} ${t}`);
  }
  return await r.json();
}

async function loadSessions() {
  const data = await fetchJson("/api/sessions?limit=50");
  sessions = data.sessions || [];
  renderChatList(sessions);

  // If no active session yet, pick the newest. If none exist, create a fresh client id.
  if (!activeSessionId) {
    activeSessionId = sessions[0]?.session_id || newSessionId();
  }

  // If active session exists in the list, load it; otherwise start empty (new session)
  if (sessions.some((s) => s.session_id === activeSessionId)) {
    await openSession(activeSessionId);
  } else {
    activeMessages = [];
    setHeaderTitle();
    renderMessages(activeMessages);
  }
}

async function openSession(sessionId) {
  activeSessionId = sessionId;

  // If the session is not yet in DynamoDB (brand-new), just reset UI.
  if (!sessions.some((s) => s.session_id === sessionId)) {
    activeMessages = [];
    setHeaderTitle();
    renderChatList(sessions);
    renderMessages(activeMessages);
    inputEl?.focus();
    return;
  }

  const data = await fetchJson(
    `/api/sessions/${encodeURIComponent(sessionId)}?limit=200`
  );

  // data.messages is already [{role, content, ts?}] via lc_messages_to_dicts
  activeMessages = data.messages || [];

  setHeaderTitle();
  renderChatList(sessions);
  renderMessages(activeMessages);

  autosizeTextarea();
  inputEl?.focus();
}

function updateLastAssistantBubble(acc) {
  // update the last assistant bubble in-place
  const last = messagesEl.lastElementChild;
  const inner = last?.querySelector?.(".cg-bubble");
  if (inner) inner.innerHTML = esc(acc).replaceAll("\n", "<br/>");
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function streamAnswer(userText) {
  const url = `/chat/stream?message=${encodeURIComponent(
    userText
  )}&session_id=${encodeURIComponent(activeSessionId)}`;
  const es = new EventSource(url);

  let acc = "";

  es.addEventListener("error", () => {
    es.close();
    acc += "\n[stream error]";
    // reflect in UI
    const lastIdx = activeMessages.length - 1;
    if (lastIdx >= 0 && activeMessages[lastIdx].role === "assistant") {
      activeMessages[lastIdx].content = acc;
    }
    updateLastAssistantBubble(acc);
  });

  es.addEventListener("end", async () => {
    es.close();

    // On completion, refresh sidebar list so the session appears / title updates
    try {
      await loadSessions();
    } catch {
      // non-fatal
    }
  });

  es.onmessage = (evt) => {
    const chunk = (evt.data || "").replaceAll("\\n", "\n");
    acc += chunk;

    // update in-memory
    const lastIdx = activeMessages.length - 1;
    if (lastIdx >= 0 && activeMessages[lastIdx].role === "assistant") {
      activeMessages[lastIdx].content = acc;
    }

    // paint incrementally
    updateLastAssistantBubble(acc);
  };
}

function sendCurrent() {
  const txt = (inputEl?.value || "").trim();
  if (!txt) return;

  inputEl.value = "";
  autosizeTextarea();

  // optimistic UI append
  activeMessages = activeMessages || [];
  activeMessages.push({ role: "user", content: txt, ts: nowIso() });
  activeMessages.push({ role: "assistant", content: "", ts: nowIso() });

  // render immediately
  setHeaderTitle();
  renderMessages(activeMessages);

  // stream server response (server persists to DynamoDB)
  streamAnswer(txt);
}

newChatBtn?.addEventListener("click", async () => {
  // Create a new client-side session id and reset UI.
  // It will appear in DynamoDB after the first message is sent.
  activeSessionId = newSessionId();
  activeMessages = [];
  setHeaderTitle();
  renderChatList(sessions);
  renderMessages(activeMessages);
  inputEl?.focus();
});

sendBtn?.addEventListener("click", sendCurrent);

inputEl?.addEventListener("input", autosizeTextarea);

inputEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendCurrent();
  }
});

(async () => {
  autosizeTextarea();
  await loadSessions();
})();
