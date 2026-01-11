const el = (id) => document.getElementById(id);

const chatListEl = el("chatList");
const chatTitleEl = el("chatTitle");
const mobileTitleEl = el("mobileTitle");
const messagesEl = el("chatMessages");
const inputEl = el("chatInput");
const sendBtn = el("sendBtn");
const newChatBtn = el("newChatBtn");

// Sidebar + controls
const sidebarEl = el("sidebar");
const sidebarToggleEl = el("sidebarToggle");
const sidebarToggleDesktopEl = el("sidebarToggleDesktop");
const sidebarCloseEl = el("sidebarClose");
const sidebarBackdropEl = el("sidebarBackdrop");
const chatSearchEl = el("chatSearch");

// Projects
const newProjectBtn = el("newProjectBtn");
const projectListEl = el("projectList");

// Settings modal
const settingsModalBtn = el("settingsModalBtn");
const settingsModalEl = el("settingsModal");
const modelSelectEl = el("modelSelect");
const accentPickerEl = el("accentPicker");
const toggleBgPickerEl = el("toggleBgPicker");
const chatBgPickerEl = el("chatBgPicker");
const scheduleTaskBtnEl = el("scheduleTaskBtn");
const taskListEl = el("taskList");

const themeLabelEl = el("themeLabel");

// Local settings
const LS_THEME = "llm_code_theme";            // legacy: slate|blue|emerald
// Appearance is forced to light mode (no user option)
const LS_APPEARANCE = "llm_code_appearance";  // legacy; no longer used
const LS_BG = "llm_code_bg";                  // legacy: matching|gray

// New preferences (color wheels)
const LS_ACCENT = "llm_code_accent_hex";      // e.g. #a281ee
const LS_TOGGLE_BG = "llm_code_toggle_bg_hex";// e.g. #eef5ff
const LS_CHAT_BG = "llm_code_chat_bg_hex";    // e.g. #ffffff

// Projects + chat metadata
const LS_PROJECTS = "llm_code_projects_v1";   // [{id,name,created_ts}]
const LS_SESSION_META = "llm_code_session_meta_v1"; // {session_id:{titleOverride?,projectId?,pinned?,deleted?}}

let systemMq = null;

// In-memory UI state (sessions + active session)
let sessions = []; // [{session_id,last_ts,title}]
let activeSessionId = "";
let activeMessages = []; // [{role, content, ts?}]
let selectedProjectId = "all"; // "all" or a project id

function loadProjects() {
  try {
    const arr = JSON.parse(localStorage.getItem(LS_PROJECTS) || "[]");
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function saveProjects(arr) {
  localStorage.setItem(LS_PROJECTS, JSON.stringify(arr || []));
}

function loadSessionMeta() {
  try {
    const obj = JSON.parse(localStorage.getItem(LS_SESSION_META) || "{}");
    return obj && typeof obj === "object" ? obj : {};
  } catch {
    return {};
  }
}

function saveSessionMeta(obj) {
  localStorage.setItem(LS_SESSION_META, JSON.stringify(obj || {}));
}

function getMeta(sessionId) {
  const meta = loadSessionMeta();
  return meta[sessionId] || {};
}

function setMeta(sessionId, patch) {
  const meta = loadSessionMeta();
  meta[sessionId] = { ...(meta[sessionId] || {}), ...(patch || {}) };
  saveSessionMeta(meta);
}

function displayTitleForSession(s) {
  const m = getMeta(s.session_id);
  const title = (m.titleOverride || s.title || "Untitled").trim() || "Untitled";
  return title;
}

function renderChatList(list) {
  chatListEl.innerHTML = "";

  const q = (chatSearchEl?.value || "").trim().toLowerCase();
  const projects = loadProjects();

  const filtered = (list || [])
    .map((s) => {
      const m = getMeta(s.session_id);
      return { ...s, _meta: m };
    })
    .filter((s) => !s._meta.deleted)
    .filter((s) => {
      if (selectedProjectId === "all") return true;
      return (s._meta.projectId || "") === selectedProjectId;
    })
    .filter((s) => {
      if (!q) return true;
      const t = displayTitleForSession(s).toLowerCase();
      return t.includes(q);
    });

  filtered
    .slice()
    .sort((a, b) => {
      const ap = a._meta.pinned ? 1 : 0;
      const bp = b._meta.pinned ? 1 : 0;
      if (ap !== bp) return bp - ap;
      return (b.last_ts || 0) - (a.last_ts || 0);
    })
    .forEach((s) => {
      const row = document.createElement("div");
      row.className = "group flex items-center gap-2";

      const btn = document.createElement("button");
      const isActive = s.session_id === activeSessionId;
      btn.className =
        "flex-1 truncate rounded-lg px-3 py-2 text-left text-sm border transition " +
        (isActive ? "chatitem-active" : "bg-white text-slate-900 border-slate-200 hover:bg-slate-50");
      btn.type = "button";

      const title = displayTitleForSession(s);
      btn.textContent = title;
      btn.addEventListener("click", () => openSession(s.session_id));

      // 3-dots menu button (shown on hover)
      const menuBtn = document.createElement("button");
      menuBtn.type = "button";
      menuBtn.className =
        "chatMenuBtn hidden group-hover:inline-flex shrink-0 h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50";
      menuBtn.textContent = "⋯";

      const menu = document.createElement("div");
      menu.className = "chatMenu hidden fixed z-[9999] w-56 rounded-xl border border-slate-200 bg-white p-1 shadow";

      const wrap = document.createElement("div");
      wrap.className = "relative";

      function item(label, onClick, extraClass = "") {
        const b = document.createElement("button");
        b.type = "button";
        b.className =
          "w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50 " + extraClass;
        b.textContent = label;
        b.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          menu.classList.add("hidden");
          onClick();
        });
        return b;
      }

      menu.appendChild(
        item("Rename", () => {
          const cur = displayTitleForSession(s);
          const next = prompt("Rename chat", cur);
          if (next === null) return;
          setMeta(s.session_id, { titleOverride: (next || "").trim() || cur });
          renderChatList(sessions);
          setHeaderTitle();
        })
      );

      // Move to project (simple prompt dropdown later)
      menu.appendChild(
        item("Move to project…", () => {
          if (!projects.length) {
            alert("No projects yet. Create one first.");
            return;
          }
          const names = projects.map((p) => p.name).join("\n");
          const pick = prompt("Move to which project? Type exact name:\n" + names);
          if (!pick) return;
          const p = projects.find((x) => x.name === pick);
          if (!p) {
            alert("Project not found.");
            return;
          }
          setMeta(s.session_id, { projectId: p.id });
          // refresh UI so it moves immediately under current filter
          renderProjects();
          renderChatList(sessions);
        })
      );

      menu.appendChild(
        item(s._meta.pinned ? "Unpin" : "Pin", () => {
          setMeta(s.session_id, { pinned: !s._meta.pinned });
          renderChatList(sessions);
        })
      );

      menu.appendChild(
        item("Delete", () => {
          const ok = confirm("Delete this chat from the sidebar? (This does not delete from DynamoDB yet.)");
          if (!ok) return;
          setMeta(s.session_id, { deleted: true });
          // If deleting active session, switch to newest remaining
          if (s.session_id === activeSessionId) {
            const remaining = sessions.filter((x) => !getMeta(x.session_id).deleted);
            activeSessionId = remaining[0]?.session_id || newSessionId();
            openSession(activeSessionId);
          }
          renderChatList(sessions);
        }, "text-red-600")
      );

      wrap.appendChild(menuBtn);

      menuBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();

        document.querySelectorAll(".chatMenu").forEach((m) => m.classList.add("hidden"));
        document.querySelectorAll(".chatMenuSub").forEach((m) => m.classList.add("hidden"));

        if (menu.parentElement !== document.body) document.body.appendChild(menu);

        menu.classList.toggle("hidden");

        const r = menuBtn.getBoundingClientRect();
        const left = Math.min(window.innerWidth - 260, r.left);
        const top = Math.min(window.innerHeight - 220, r.bottom + 6);
        menu.style.left = left + "px";
        menu.style.top = top + "px";
      });


      row.appendChild(btn);
      row.appendChild(wrap);
      chatListEl.appendChild(row);
    });
}

function autosizeTextarea() {
  if (!inputEl) return;
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

function setHeaderTitle() {
  const found = sessions.find((s) => s.session_id === activeSessionId);
  const title = found ? displayTitleForSession(found) : "New chat";
  if (chatTitleEl) chatTitleEl.textContent = title;
  if (mobileTitleEl) mobileTitleEl.textContent = title;
}

async function fetchJson(url) {
  const r = await fetch(url, { credentials: "same-origin" });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText} ${t}`);
  }
  return await r.json();
}

function newSessionId() {
  // simple client-generated id; server treats it as an opaque string
  // (we'll migrate to server-issued ids later if desired)
  return "s_" + Date.now() + "_" + Math.random().toString(16).slice(2);
}

function bubble(role, content) {
  const wrap = document.createElement("div");
  const isUser = role === "user";

  wrap.className = "flex w-full " + (isUser ? "justify-end" : "justify-start");

  const inner = document.createElement("div");
  inner.className =
    "max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed shadow-sm border " +
    (isUser
      ? "bubble-user"
      : "bg-white text-slate-900 border-slate-200");

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
  const inner = last?.firstElementChild;
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
  messagesEl.scrollTop = messagesEl.scrollHeight;

  // stream server response (server persists to DynamoDB)
  streamAnswer(txt);
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

function renderProjects() {
  if (!projectListEl) return;
  const projects = loadProjects().slice().sort((a, b) => (b.created_ts || 0) - (a.created_ts || 0));
  projectListEl.innerHTML = "";

  function addBtn(label, id) {
    const b = document.createElement("button");
    b.type = "button";
    const active = selectedProjectId === id;
    b.className =
      "w-full truncate rounded-lg px-3 py-2 text-left text-sm border transition " +
      (active ? "chatitem-active" : "bg-white text-slate-900 border-slate-200 hover:bg-slate-50");
    b.textContent = label;
    b.addEventListener("click", () => {
      selectedProjectId = id;
      renderProjects();
      renderChatList(sessions);
    });
    projectListEl.appendChild(b);
  }

  addBtn("All", "all");
  projects.forEach((p) => addBtn(p.name, p.id));
}

function createProject() {
  const name = prompt("New project name");
  if (!name) return null;
  const trimmed = name.trim();
  if (!trimmed) return null;
  const projects = loadProjects();
  const id = "p_" + Date.now() + "_" + Math.random().toString(16).slice(2);
  const p = { id, name: trimmed, created_ts: Date.now() };
  projects.unshift(p);
  saveProjects(projects);
  renderProjects();
  return p;
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

newProjectBtn?.addEventListener("click", (e) => {
  e.preventDefault();
  createProject();
});

chatSearchEl?.addEventListener("input", () => {
  renderChatList(sessions);
});

sendBtn?.addEventListener("click", sendCurrent);

inputEl?.addEventListener("input", autosizeTextarea);

inputEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendCurrent();
  }
});

function setActiveChoice(buttons, predicate) {
  (buttons || []).forEach((b) => {
    const on = predicate(b);
    b.setAttribute("aria-pressed", on ? "true" : "false");
    b.classList.toggle("ring-2", on);
    b.classList.toggle("ring-slate-400", on);
  });
}

function getSystemMode() {
  return "light";
}

function applyAppearance(_pref) {
  const mode = "light";
  document.documentElement.dataset.appearance = "light";
  document.documentElement.dataset.mode = mode;
  document.documentElement.style.colorScheme = mode;

  // remove any previous system listener
  if (systemMq) {
    try { systemMq.onchange = null; } catch {}
    systemMq = null;
  }

  // ensure any leftover UI buttons don't show as active
  setActiveChoice(document.querySelectorAll(".appearanceOpt"), () => false);
}

function applyBackground(bg) {
  const b = (bg || "").trim() || "matching";
  try { localStorage.setItem(LS_BG, b); } catch {}
  document.documentElement.dataset.bg = b;
  setActiveChoice(document.querySelectorAll(".bgOpt"), (x) => (x.dataset.bg || "") === b);
}

function applyTheme(t) {
  const theme = (t || "").trim() || "slate";
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(LS_THEME, theme);
  } catch {}
  if (themeLabelEl) {
    themeLabelEl.textContent = theme[0].toUpperCase() + theme.slice(1);
  }
  setActiveChoice(document.querySelectorAll(".themeOpt"), (b) => (b.dataset.theme || "") === theme);
}

function openSidebar() {
  if (!sidebarEl) return;
  sidebarEl.classList.remove("hidden");
  sidebarEl.classList.add("flex");
  sidebarBackdropEl?.classList.remove("hidden");
}

function closeSidebar() {
  if (!sidebarEl) return;
  // only hide on mobile
  if (window.matchMedia && window.matchMedia("(min-width: 768px)").matches) return;
  sidebarEl.classList.add("hidden");
  sidebarEl.classList.remove("flex");
  sidebarBackdropEl?.classList.add("hidden");
}

function initSidebarControls() {
  sidebarToggleEl?.addEventListener("click", (e) => {
    e.preventDefault();
    openSidebar();
  });
  sidebarCloseEl?.addEventListener("click", (e) => {
    e.preventDefault();
    closeSidebar();
  });
  sidebarBackdropEl?.addEventListener("click", () => closeSidebar());

  sidebarToggleDesktopEl?.addEventListener("click", (e) => {
    e.preventDefault();
    sidebarEl?.classList.toggle("sidebar-collapsed");
    document.body.classList.toggle(
      "sidebar-collapsed",
      sidebarEl?.classList.contains("sidebar-collapsed")
    );
  });
  
  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t && (t.closest?.(".chatMenu") || t.closest?.(".chatMenuSub") || t.closest?.(".chatMenuBtn"))) return;
    document.querySelectorAll(".chatMenu").forEach((m) => m.classList.add("hidden"));
    document.querySelectorAll(".chatMenuSub").forEach((m) => m.classList.add("hidden"));
  });
}

function openSettingsModal() {
  if (!settingsModalEl) return;
  settingsModalEl.classList.remove("hidden");
  settingsModalEl.setAttribute("aria-hidden", "false");
}

function closeSettingsModal() {
  if (!settingsModalEl) return;
  settingsModalEl.classList.add("hidden");
  settingsModalEl.setAttribute("aria-hidden", "true");
}

function initSettingsModal() {
  // initialize legacy settings (theme/appearance/bg)
  const curTheme = (() => { try { return localStorage.getItem(LS_THEME) || "slate"; } catch { return "slate"; } })();
  const curAppearance = "light";
  const curBg = (() => { try { return localStorage.getItem(LS_BG) || "matching"; } catch { return "matching"; } })();
  applyTheme(curTheme);
  applyAppearance(curAppearance);
  applyBackground(curBg);

  // initialize new color pickers
  const accent = (() => { try { return localStorage.getItem(LS_ACCENT) || "#a281ee"; } catch { return "#a281ee"; } })();
  const toggleBg = (() => { try { return localStorage.getItem(LS_TOGGLE_BG) || "#eef5ff"; } catch { return "#eef5ff"; } })();
  const chatBg = (() => { try { return localStorage.getItem(LS_CHAT_BG) || "#ffffff"; } catch { return "#ffffff"; } })();
  if (accentPickerEl) accentPickerEl.value = accent;
  if (toggleBgPickerEl) toggleBgPickerEl.value = toggleBg;
  if (chatBgPickerEl) chatBgPickerEl.value = chatBg;

  // Apply CSS variables (CSS will use these later)
  document.documentElement.style.setProperty("--accent", accent);
  document.documentElement.style.setProperty("--toggle-bg", toggleBg);
  document.documentElement.style.setProperty("--chat-bg", chatBg);

  settingsModalBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    openSettingsModal();
  });

  // close buttons/overlay
  settingsModalEl?.querySelectorAll('[data-close="settings"]').forEach((x) => {
    x.addEventListener("click", (e) => {
      e.preventDefault();
      closeSettingsModal();
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSettingsModal();
  });

  // color pickers persistence
  accentPickerEl?.addEventListener("input", () => {
    const v = accentPickerEl.value;
    try { localStorage.setItem(LS_ACCENT, v); } catch {}
    document.documentElement.style.setProperty("--accent", v);
  });

  toggleBgPickerEl?.addEventListener("input", () => {
    const v = toggleBgPickerEl.value;
    try { localStorage.setItem(LS_TOGGLE_BG, v); } catch {}
    document.documentElement.style.setProperty("--toggle-bg", v);

    // toggle-bg only drives the chat surface when Background = gray
    const cur = document.documentElement.dataset.bg || "matching";
    if (cur !== "gray") {
      applyBackground("gray");
      try { localStorage.setItem(LS_BG, "gray"); } catch {}
      if (themeLabelEl) themeLabelEl.textContent = "Gray";
    }
  });
  chatBgPickerEl?.addEventListener("input", () => {
    const v = chatBgPickerEl.value;
    try { localStorage.setItem(LS_CHAT_BG, v); } catch {}
    document.documentElement.style.setProperty("--chat-bg", v);
  });

  // Model dropdown placeholder (backend wiring later)
  modelSelectEl?.addEventListener("change", () => {
    // store selection for later
    try { localStorage.setItem("llm_code_model", modelSelectEl.value); } catch {}
  });

  // Tasks placeholder (backend wiring later)
  scheduleTaskBtnEl?.addEventListener("click", (e) => {
    e.preventDefault();
    alert("Task scheduling UI coming next.");
  });
}

(async () => {
  autosizeTextarea();
  initSidebarControls();
  initSettingsModal();
  renderProjects();
  await loadSessions();
})();
