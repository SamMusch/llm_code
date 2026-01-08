const el = (id) => document.getElementById(id);

const chatListEl = el("chatList");
const chatTitleEl = el("chatTitle");
const messagesEl = el("chatMessages");
const inputEl = el("chatInput");
const sendBtn = el("sendBtn");
const newChatBtn = el("newChatBtn");

// Sidebar hide/unhide (UI-only)
const LS_HIDDEN = "llm_code_hidden_sessions_v1";
let showHidden = false;

// In-memory UI state (sessions + active session)
let sessions = []; // [{session_id,last_ts,title}]
let activeSessionId = "";
let activeMessages = []; // [{role, content, ts?}]

function loadHiddenSet() {
  try {
    const arr = JSON.parse(localStorage.getItem(LS_HIDDEN) || "[]");
    return new Set(Array.isArray(arr) ? arr : []);
  } catch {
    return new Set();
  }
}

function saveHiddenSet(set) {
  localStorage.setItem(LS_HIDDEN, JSON.stringify(Array.from(set)));
}

function isHidden(sessionId) {
  return loadHiddenSet().has(sessionId);
}

function setHidden(sessionId, hidden) {
  const s = loadHiddenSet();
  if (hidden) s.add(sessionId);
  else s.delete(sessionId);
  saveHiddenSet(s);
}

function ensureHiddenToggleButton() {
  if (!newChatBtn) return;
  const existing = document.getElementById("toggleHiddenBtn");
  if (existing) return;

  const btn = document.createElement("button");
  btn.id = "toggleHiddenBtn";
  btn.type = "button";
  btn.className = "btn sidebar";
  btn.textContent = "Show hidden";
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    showHidden = !showHidden;
    btn.textContent = showHidden ? "Hide hidden" : "Show hidden";
    renderChatList(sessions);
  });

  // Place next to the New chat button
  newChatBtn.parentElement?.appendChild(btn);
}

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

  const hidden = loadHiddenSet();
  const visibleList = (list || []).filter((s) => showHidden || !hidden.has(s.session_id));

  visibleList
    .slice()
    .sort((a, b) => (b.last_ts || 0) - (a.last_ts || 0))
    .forEach((s) => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.gap = "8px";
      row.style.alignItems = "center";

      const btn = document.createElement("button");
      btn.className =
        "cg-chatitem" + (s.session_id === activeSessionId ? " active" : "");
      btn.type = "button";

      const title = (s.title || "Untitled").trim() || "Untitled";
      const isRowHidden = hidden.has(s.session_id);
      btn.textContent = isRowHidden ? `[hidden] ${title}` : title;

      btn.addEventListener("click", () => {
        openSession(s.session_id);
      });

      const hideBtn = document.createElement("button");
      hideBtn.type = "button";
      hideBtn.className = "btn sidebar";
      hideBtn.style.padding = "6px 10px";
      hideBtn.style.borderRadius = "10px";
      hideBtn.textContent = isRowHidden ? "Unhide" : "Hide";
      hideBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();

        const nextHidden = !hidden.has(s.session_id);
        setHidden(s.session_id, nextHidden);

        // If you hid the active session, move focus to newest visible session.
        if (nextHidden && s.session_id === activeSessionId) {
          const newestVisible = (sessions || []).filter((x) => !isHidden(x.session_id))[0];
          if (newestVisible) openSession(newestVisible.session_id);
        }

        renderChatList(sessions);
      });

      row.appendChild(btn);
      row.appendChild(hideBtn);
      chatListEl.appendChild(row);
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
  ensureHiddenToggleButton();
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
