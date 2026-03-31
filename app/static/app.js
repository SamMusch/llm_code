const $ = (id) => document.getElementById(id);
const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => [...root.querySelectorAll(sel)];
const on = (el, evt, fn) => el?.addEventListener(evt, fn);

const LS = {
  THEME: "llm_code_theme",
  APPEARANCE: "llm_code_appearance",
  BG: "llm_code_bg",
  ACCENT: "llm_code_accent_hex",
  TOGGLE_BG: "llm_code_toggle_bg_hex",
  CHAT_BG: "llm_code_chat_bg_hex",
  MODEL: "llm_code_model",
  PROJECTS: "llm_code_projects_v1",
  SESSION_META: "llm_code_session_meta_v1",
  RIGHTBAR: "llm_code_rightbar_v1",
  SELECTED_TOOLS: "llm_code_selected_tools_v1",
};

const DEFAULTS = {
  ACCENT: "#bfbfbf",
  TOGGLE_BG: "#ECECEC",
  CHAT_BG: "#FFFFFF",
  THEME: "slate",
  BG: "matching",
};

const ui = {
  chatList: $("chatList"),
  chatTitle: $("chatTitle"),
  mobileTitle: $("mobileTitle"),
  messages: $("chatMessages"),
  input: $("chatInput"),
  sendBtn: $("sendBtn"),
  newChatBtn: $("newChatBtn"),

  attachBtn: $("attachBtn"),
  fileInput: $("fileInput"),
  attachmentStrip: $("attachmentStrip"),

  sidebar: $("sidebar"),
  sidebarToggle: $("sidebarToggle"),
  sidebarToggleDesktop: $("sidebarToggleDesktop"),
  sidebarClose: $("sidebarClose"),
  sidebarBackdrop: $("sidebarBackdrop"),
  chatSearch: $("chatSearch"),

  rightbar: $("rightbar"),
  rightbarToggle: $("rightbarToggle"),
  rightbarToggleDesktop: $("rightbarToggleDesktop"),
  rightbarClose: $("rightbarClose"),
  rightbarBackdrop: $("rightbarBackdrop"),

  toolDb: $("toolDb"),
  toolPlaceholder1: $("toolPlaceholder1"),
  placeholderA: $("placeholderA"),
  placeholderB: $("placeholderB"),

  metaTitle: $("metaTitle"),
  metaTags: $("metaTags"),
  metaWordMin: $("metaWordMin"),
  metaWordMax: $("metaWordMax"),
  metaCreatedFrom: $("metaCreatedFrom"),
  metaCreatedTo: $("metaCreatedTo"),
  metaModifiedFrom: $("metaModifiedFrom"),
  metaModifiedTo: $("metaModifiedTo"),

  newProjectBtn: $("newProjectBtn"),
  projectList: $("projectList"),

  settingsModalBtn: $("settingsModalBtn"),
  settingsModal: $("settingsModal"),
  modelSelect: $("modelSelect"),
  accentHex: $("accentHex"),
  accentPicker: $("accentPicker"),
  toggleBgHex: $("toggleBgHex"),
  toggleBgPicker: $("toggleBgPicker"),
  chatBgHex: $("chatBgHex"),
  chatBgPicker: $("chatBgPicker"),
  restoreDefaultsBtn: $("restoreDefaultsBtn"),
  scheduleTaskBtn: $("scheduleTaskBtn"),
  taskList: $("taskList"),
  themeLabel: $("themeLabel"),
};

const state = {
  systemMq: null,
  pendingAttachments: [],
  sessions: [],
  activeSessionId: "",
  activeMessages: [],
  selectedProjectId: "all",
};

function loadJsonLS(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key) || "") ?? fallback;
  } catch {
    return fallback;
  }
}

function saveJsonLS(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

function getLsValue(key, fallback = "") {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : value;
  } catch {
    return fallback;
  }
}

function setLsValue(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

async function fetchJson(url) {
  const r = await fetch(url, { credentials: "same-origin" });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText} ${t}`);
  }
  return await r.json();
}

function syncDateInputState(root = document) {
  qsa('#rightbar input[type="date"]', root).forEach((input) => {
    input.classList.toggle("has-value", !!input.value);
  });
}

function loadProjects() {
  const arr = loadJsonLS(LS.PROJECTS, []);
  return Array.isArray(arr) ? arr : [];
}

function saveProjects(arr) {
  saveJsonLS(LS.PROJECTS, arr || []);
}

function loadSessionMeta() {
  const obj = loadJsonLS(LS.SESSION_META, {});
  return obj && typeof obj === "object" ? obj : {};
}

function saveSessionMeta(obj) {
  saveJsonLS(LS.SESSION_META, obj || {});
}

function getMeta(sessionId) {
  return loadSessionMeta()[sessionId] || {};
}

function setMeta(sessionId, patch) {
  const meta = loadSessionMeta();
  meta[sessionId] = { ...(meta[sessionId] || {}), ...(patch || {}) };
  saveSessionMeta(meta);
}

function displayTitleForSession(s) {
  const m = getMeta(s.session_id);
  return (m.titleOverride || s.title || "Untitled").trim() || "Untitled";
}

function autosizeTextarea() {
  if (!ui.input) return;
  ui.input.style.height = "auto";
  ui.input.style.height = Math.min(ui.input.scrollHeight, 160) + "px";
}

function setHeaderTitle() {
  const found = state.sessions.find((s) => s.session_id === state.activeSessionId);
  const title = found ? displayTitleForSession(found) : "New chat";
  if (ui.chatTitle) ui.chatTitle.textContent = title;
  if (ui.mobileTitle) ui.mobileTitle.textContent = title;
}

function newSessionId() {
  return "s_" + Date.now() + "_" + Math.random().toString(16).slice(2);
}

function bubble(role, content) {
  const wrap = document.createElement("div");
  const isUser = role === "user";
  wrap.className = "flex w-full " + (isUser ? "justify-end" : "justify-start");

  const inner = document.createElement("div");
  inner.className =
    "max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed shadow-sm border " +
    (isUser ? "bubble-user" : "bg-white text-slate-900 border-slate-200");

  const body = document.createElement("div");
  body.className = "messageBody";
  body.innerHTML = renderMarkdown(content);

  inner.appendChild(body);
  wrap.appendChild(inner);
  return { wrap, inner };
}

function renderMessages(msgs) {
  if (!ui.messages) return;
  ui.messages.innerHTML = "";
  (msgs || []).forEach((m) => {
    const { wrap } = bubble(m.role, m.content);
    ui.messages.appendChild(wrap);
  });
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function ensureResponsePanelsContainer() {
  const last = ui.messages?.lastElementChild;
  const inner = last?.firstElementChild;
  if (!inner) return null;

  let row = qs(".responsePanelsRow", inner);
  if (row) return row;

  row = document.createElement("div");
  row.className = "responsePanelsRow mt-2 flex flex-wrap gap-2";
  inner.appendChild(row);
  return row;
}

function ensureStepsPanel() {
  const row = ensureResponsePanelsContainer();
  if (!row) return null;

  let panel = qs(".stepsPanel", row);
  if (panel) return panel;

  panel = document.createElement("details");
  panel.className = "stepsPanel min-w-[220px] flex-1 rounded-lg border border-slate-200 bg-slate-50 p-2 text-xs";

  const summary = document.createElement("summary");
  summary.className = "cursor-pointer select-none text-slate-700";
  summary.textContent = "Show steps";

  const meta = document.createElement("div");
  meta.className = "stepsMeta mt-1 text-slate-600";

  const list = document.createElement("div");
  list.className = "stepsList mt-2 space-y-2";

  panel.append(summary, meta, list);
  row.appendChild(panel);
  return panel;
}

function ensureSourcesPanel() {
  const row = ensureResponsePanelsContainer();
  if (!row) return null;

  let panel = qs(".sourcesPanel", row);
  if (panel) return panel;

  panel = document.createElement("details");
  panel.className = "sourcesPanel min-w-[220px] flex-1 rounded-lg border border-slate-200 bg-slate-50 p-2 text-xs";

  const summary = document.createElement("summary");
  summary.className = "cursor-pointer select-none text-slate-700";
  summary.textContent = "Show sources";

  const body = document.createElement("div");
  body.className = "sourcesBody mt-2 text-slate-800";

  panel.append(summary, body);
  row.appendChild(panel);
  return panel;
}

function setStepsMeta(traceId) {
  const panel = ensureStepsPanel();
  if (!panel) return;
  const meta = qs(".stepsMeta", panel);
  if (meta) meta.textContent = traceId ? `trace_id: ${traceId}` : "";
}

function asStepText(v) {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function renderSteps(steps) {
  const panel = ensureStepsPanel();
  if (!panel) return;
  const list = qs(".stepsList", panel);
  if (!list) return;
  list.innerHTML = "";

  (steps || []).forEach((s) => {
    const name = s?.name || "";
    const status = s?.status || "";
    const combinedText = (
      asStepText(s?.input) + asStepText(s?.output) + asStepText(s?.error)
    ).trim();

    if ((s?.step_type === "step" || !s?.step_type) && !name && !status && !combinedText) return;

    const row = document.createElement("div");
    row.className = "rounded-md border border-slate-200 bg-white p-2";

    const head = document.createElement("div");
    head.className = "font-medium text-slate-900";
    head.textContent = `${s.step_type || "step"}: ${name} — ${status}`;

    const pre = document.createElement("pre");
    pre.className = "mt-1 whitespace-pre-wrap text-slate-800";

    if (s.status === "start") {
      const t = asStepText(s.input);
      pre.textContent = t ? `input:\n${t}` : "";
    } else if (s.status === "ok") {
      const t = asStepText(s.output);
      pre.textContent = t ? `output:\n${t}` : "";
    } else {
      const t = asStepText(s.error);
      pre.textContent = t ? `error:\n${t}` : "";
    }

    row.appendChild(head);
    if (pre.textContent) row.appendChild(pre);
    list.appendChild(row);
  });
}

function renderSources(sources) {
  const panel = ensureSourcesPanel();
  if (!panel) return;
  const body = qs(".sourcesBody", panel);
  if (!body) return;

  body.innerHTML = "";

  const items = Array.isArray(sources) ? sources : [];
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "text-slate-600";
    empty.textContent = "No sources available.";
    body.appendChild(empty);
    return;
  }

  const title = document.createElement("div");
  title.className = "mb-2 font-medium text-slate-900";
  title.textContent = "Sources:";
  body.appendChild(title);

  const ol = document.createElement("ol");
  ol.className = "list-decimal pl-5 space-y-1";
  body.appendChild(ol);

  items.forEach((src) => {
    const li = document.createElement("li");
    const hasUrl = typeof src?.webUrl === "string" && src.webUrl.trim();
    const label = String(src?.name || src?.source || "unknown").trim() || "unknown";
    const node = hasUrl ? document.createElement("a") : document.createElement("span");
    node.textContent = label;

    if (hasUrl) {
      node.href = src.webUrl;
      node.target = "_blank";
      node.rel = "noopener noreferrer";
      node.className = "underline hover:no-underline";
    }

    li.appendChild(node);

    const pageNumber = src?.page_number;
    if (pageNumber !== null && pageNumber !== undefined && pageNumber !== "") {
      li.appendChild(document.createTextNode(` - slide ${pageNumber}`));
    }

    ol.appendChild(li);
  });
}

function renderChatList(list) {
  if (!ui.chatList) return;
  ui.chatList.innerHTML = "";

  const q = (ui.chatSearch?.value || "").trim().toLowerCase();
  const projects = loadProjects();

  const filtered = (list || [])
    .map((s) => ({ ...s, _meta: getMeta(s.session_id) }))
    .filter((s) => !s._meta.deleted)
    .filter((s) => state.selectedProjectId === "all" || (s._meta.projectId || "") === state.selectedProjectId)
    .filter((s) => !q || displayTitleForSession(s).toLowerCase().includes(q));

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
      const isActive = s.session_id === state.activeSessionId;
      btn.className =
        "flex-1 truncate rounded-lg px-3 py-2 text-left text-sm border transition " +
        (isActive ? "chatitem-active" : "bg-white text-slate-900 border-slate-200 hover:bg-slate-50");
      btn.type = "button";
      btn.textContent = displayTitleForSession(s);
      btn.addEventListener("click", () => openSession(s.session_id));

      const menuBtn = document.createElement("button");
      menuBtn.type = "button";
      menuBtn.className =
        "chatMenuBtn hidden group-hover:inline-flex shrink-0 h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50";
      menuBtn.textContent = "⋯";

      const menu = document.createElement("div");
      menu.className = "chatMenu hidden fixed z-[9999] w-56 rounded-xl border border-slate-200 bg-white p-1 shadow";

      const wrap = document.createElement("div");
      wrap.className = "relative";

      function menuItem(label, onClick, extraClass = "") {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50 ${extraClass}`;
        b.textContent = label;
        b.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          menu.classList.add("hidden");
          onClick();
        });
        return b;
      }

      menu.appendChild(menuItem("Rename", () => {
        const cur = displayTitleForSession(s);
        const next = prompt("Rename chat", cur);
        if (next === null) return;
        setMeta(s.session_id, { titleOverride: (next || "").trim() || cur });
        renderChatList(state.sessions);
        setHeaderTitle();
      }));

      menu.appendChild(menuItem("Move to project…", () => {
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
        renderProjects();
        renderChatList(state.sessions);
      }));

      menu.appendChild(menuItem(s._meta.pinned ? "Unpin" : "Pin", () => {
        setMeta(s.session_id, { pinned: !s._meta.pinned });
        renderChatList(state.sessions);
      }));

      menu.appendChild(menuItem("Delete", () => {
        const ok = confirm("Delete this chat from the sidebar? (This does not delete from DynamoDB yet.)");
        if (!ok) return;
        setMeta(s.session_id, { deleted: true });
        if (s.session_id === state.activeSessionId) {
          const remaining = state.sessions.filter((x) => !getMeta(x.session_id).deleted);
          state.activeSessionId = remaining[0]?.session_id || newSessionId();
          openSession(state.activeSessionId);
        }
        renderChatList(state.sessions);
      }, "text-red-600"));

      wrap.appendChild(menuBtn);

      menuBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();

        qsa(".chatMenu").forEach((m) => m.classList.add("hidden"));
        qsa(".chatMenuSub").forEach((m) => m.classList.add("hidden"));

        if (menu.parentElement !== document.body) document.body.appendChild(menu);
        menu.classList.toggle("hidden");

        const r = menuBtn.getBoundingClientRect();
        menu.style.left = Math.min(window.innerWidth - 260, r.left) + "px";
        menu.style.top = Math.min(window.innerHeight - 220, r.bottom + 6) + "px";
      });

      row.append(btn, wrap);
      ui.chatList.appendChild(row);
    });
}

async function refreshSessionsListOnly() {
  const data = await fetchJson("/app/api/sessions?limit=50");
  state.sessions = data.sessions || [];
  renderChatList(state.sessions);
}

async function loadSessions() {
  const data = await fetchJson("/app/api/sessions?limit=50");
  state.sessions = data.sessions || [];
  renderChatList(state.sessions);

  if (!state.activeSessionId) {
    state.activeSessionId = state.sessions[0]?.session_id || newSessionId();
  }

  if (state.sessions.some((s) => s.session_id === state.activeSessionId)) {
    await openSession(state.activeSessionId);
  } else {
    state.activeMessages = [];
    setHeaderTitle();
    renderMessages(state.activeMessages);
  }
}

async function openSession(sessionId) {
  state.activeSessionId = sessionId;

  if (!state.sessions.some((s) => s.session_id === sessionId)) {
    state.activeMessages = [];
    setHeaderTitle();
    renderChatList(state.sessions);
    renderMessages(state.activeMessages);
    ui.input?.focus();
    return;
  }

  const data = await fetchJson(`/app/api/sessions/${encodeURIComponent(sessionId)}?limit=200`);
  state.activeMessages = data.messages || [];

  setHeaderTitle();
  renderChatList(state.sessions);
  renderMessages(state.activeMessages);
  autosizeTextarea();
  ui.input?.focus();
}

function updateLastAssistantBubble(acc) {
  const last = ui.messages?.lastElementChild;
  const inner = last?.firstElementChild;
  const body = inner?.querySelector?.(".messageBody") || inner;
  if (body) body.innerHTML = renderMarkdown(acc);
  if (ui.messages) ui.messages.scrollTop = ui.messages.scrollHeight;
}

function _csvToList(s) {
  return (s || "").split(",").map((x) => x.trim()).filter(Boolean);
}

function buildMetadataFilters() {
  const filters = {};
  const title = (ui.metaTitle?.value || "").trim();
  if (title) filters.title = title;

  const tags = _csvToList(ui.metaTags?.value || "");
  if (tags.length) filters.tags = tags;

  return filters;
}

function buildFiltersJson() {
  const out = {};
  out.TOOLS = getSelectedTools().slice();
  out.PLACEHOLDER = [];

  const metaList = [];
  Object.entries(buildMetadataFilters()).forEach(([k, v]) => {
    if (v == null) return;
    if (Array.isArray(v)) {
      if (v.length) metaList.push(`${k}=${v.join(",")}`);
    } else {
      const s = String(v).trim();
      if (s) metaList.push(`${k}=${s}`);
    }
  });
  out.METADATA = metaList;

  return out;
}

function getSelectedTools() {
  const arr = loadJsonLS(LS.SELECTED_TOOLS, []);
  return Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : [];
}

function setSelectedTools(arr) {
  saveJsonLS(LS.SELECTED_TOOLS, Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : []);
}

function streamAnswer(userText, fileIds = []) {
  const ids = Array.isArray(fileIds) ? fileIds.filter(Boolean) : [];
  const fileParam = ids.length ? `&file_ids=${encodeURIComponent(ids.join(","))}` : "";

  const tools = getSelectedTools();
  const toolsParam = tools.length ? `&tools=${encodeURIComponent(tools.join(","))}` : "";

  const metaFilters = buildMetadataFilters();
  const filtersParam = Object.keys(metaFilters).length
    ? `&filters=${encodeURIComponent(JSON.stringify(metaFilters))}`
    : "";

  const filtersJson = buildFiltersJson();
  const filtersJsonParam = Object.keys(filtersJson).length
    ? `&filters_json=${encodeURIComponent(JSON.stringify(filtersJson))}`
    : "";

  let projectId = "";
  let projectName = "";
  try {
    const m = getMeta(state.activeSessionId);
    projectId = (m.projectId || "").trim();
    if (projectId) {
      const p = loadProjects().find((x) => (x.id || "") === projectId);
      projectName = (p?.name || "").trim();
    }
  } catch {}

  const projectIdParam = projectId ? `&project_id=${encodeURIComponent(projectId)}` : "";
  const projectNameParam = projectName ? `&project_name=${encodeURIComponent(projectName)}` : "";

  const model = (getLsValue(LS.MODEL, "") || "").trim();
  const modelParam = model ? `&model=${encodeURIComponent(model)}` : "";

  const url = `/app/chat/stream?message=${encodeURIComponent(userText)}&session_id=${encodeURIComponent(state.activeSessionId)}${fileParam}${toolsParam}${filtersParam}${filtersJsonParam}${projectIdParam}${projectNameParam}${modelParam}`;
  const es = new EventSource(url);

  let acc = "";
  let steps = [];
  let traceId = "";

  es.addEventListener("error", () => {
    es.close();
    acc += "\n[stream error]";
    const lastIdx = state.activeMessages.length - 1;
    if (lastIdx >= 0 && state.activeMessages[lastIdx].role === "assistant") {
      state.activeMessages[lastIdx].content = acc;
    }
    updateLastAssistantBubble(acc);
  });

  es.addEventListener("meta", (evt) => {
    try {
      const obj = JSON.parse(evt.data || "{}");
      traceId = obj.trace_id || "";
      setStepsMeta(traceId);
    } catch {}
  });

  es.addEventListener("sources", (evt) => {
    try {
      renderSources(JSON.parse(evt.data || "[]"));
    } catch {}
  });

  es.addEventListener("token", (evt) => {
    const chunk = (evt.data || "").replaceAll("\\n", "\n");
    acc += chunk;
    const lastIdx = state.activeMessages.length - 1;
    if (lastIdx >= 0 && state.activeMessages[lastIdx].role === "assistant") {
      state.activeMessages[lastIdx].content = acc;
    }
    updateLastAssistantBubble(acc);
  });

  es.addEventListener("step", (evt) => {
    try {
      steps.push(JSON.parse(evt.data || "{}"));
      renderSteps(steps);
    } catch {}
  });

  es.onmessage = (evt) => {
    const chunk = (evt.data || "").replaceAll("\\n", "\n");
    if (!chunk) return;
    acc += chunk;
    const lastIdx = state.activeMessages.length - 1;
    if (lastIdx >= 0 && state.activeMessages[lastIdx].role === "assistant") {
      state.activeMessages[lastIdx].content = acc;
    }
    updateLastAssistantBubble(acc);
  };

  es.addEventListener("end", async () => {
    es.close();
    try {
      await refreshSessionsListOnly();
    } catch {}
  });
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

function renderMarkdown(md) {
  let s = esc(md || "");
  s = s.replace(/```([\s\S]*?)```/g, (m, code) => {
    const c = code.replace(/^[\n\r]+|[\n\r]+$/g, "");
    return `<pre class="whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 p-3 overflow-x-auto"><code>${c}</code></pre>`;
  });
  s = s.replace(/`([^`]+?)`/g, (m, code) => `<code class="rounded bg-slate-100 px-1 py-0.5">${code}</code>`);
  s = s.replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*]+?)\*(?!\*)/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="underline">$1</a>');
  return s.replace(/\n/g, "<br/>");
}

function renderAttachmentStrip() {
  if (!ui.attachmentStrip) return;
  const list = state.pendingAttachments || [];
  if (!list.length) {
    ui.attachmentStrip.classList.add("hidden");
    ui.attachmentStrip.innerHTML = "";
    return;
  }

  ui.attachmentStrip.classList.remove("hidden");
  ui.attachmentStrip.innerHTML = "";

  list.forEach((f) => {
    const chip = document.createElement("div");
    chip.className = "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-900";

    const name = document.createElement("span");
    name.className = "max-w-[220px] truncate";
    name.textContent = f.name || "file";

    const x = document.createElement("button");
    x.type = "button";
    x.className = "h-5 w-5 rounded-full border border-slate-200 bg-white text-slate-700 hover:bg-slate-50";
    x.textContent = "×";
    x.setAttribute("aria-label", "Remove attachment");
    x.addEventListener("click", () => {
      state.pendingAttachments = state.pendingAttachments.filter((a) => a.id !== f.id);
      renderAttachmentStrip();
    });

    chip.append(name, x);
    ui.attachmentStrip.appendChild(chip);
  });
}

async function uploadSelectedFiles(fileList) {
  const files = Array.from(fileList || []).filter(Boolean);
  if (!files.length) return;

  const fd = new FormData();
  files.forEach((file) => fd.append("files", file, file.name));

  const url = `/app/api/files?session_id=${encodeURIComponent(state.activeSessionId)}`;
  const r = await fetch(url, { method: "POST", body: fd, credentials: "same-origin" });

  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText} ${t}`);
  }

  const data = await r.json();
  const returned = Array.isArray(data.files) ? data.files : [];
  const byId = new Map((state.pendingAttachments || []).map((x) => [x.id, x]));

  returned.forEach((f) => {
    if (!f || !f.id) return;
    byId.set(f.id, { id: f.id, name: f.name || "file", size: f.size || 0 });
  });

  state.pendingAttachments = Array.from(byId.values());
  renderAttachmentStrip();
}

function renderProjects() {
  if (!ui.projectList) return;
  const projects = loadProjects().slice().sort((a, b) => (b.created_ts || 0) - (a.created_ts || 0));
  ui.projectList.innerHTML = "";

  function addBtn(label, id) {
    const b = document.createElement("button");
    b.type = "button";
    const active = state.selectedProjectId === id;
    b.className =
      "w-full truncate rounded-lg px-3 py-2 text-left text-sm border transition " +
      (active ? "chatitem-active" : "bg-white text-slate-900 border-slate-200 hover:bg-slate-50");
    b.textContent = label;
    b.addEventListener("click", () => {
      state.selectedProjectId = id;
      renderProjects();
      renderChatList(state.sessions);
    });
    ui.projectList.appendChild(b);
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

function sendCurrent() {
  const txt = (ui.input?.value || "").trim();
  if (!txt) return;

  const fileIds = state.pendingAttachments.map((x) => x.id).filter(Boolean);
  if (ui.input) ui.input.value = "";
  autosizeTextarea();

  state.activeMessages.push({ role: "user", content: txt, ts: nowIso() });
  state.activeMessages.push({ role: "assistant", content: "", ts: nowIso() });

  setHeaderTitle();
  renderMessages(state.activeMessages);
  if (ui.messages) ui.messages.scrollTop = ui.messages.scrollHeight;

  state.pendingAttachments = [];
  renderAttachmentStrip();
  streamAnswer(txt, fileIds);
}

function setActiveChoice(buttons, predicate) {
  (buttons || []).forEach((b) => {
    const onState = predicate(b);
    b.setAttribute("aria-pressed", onState ? "true" : "false");
    b.classList.toggle("ring-2", onState);
    b.classList.toggle("ring-slate-400", onState);
  });
}

function applyAppearance(_pref) {
  const mode = "light";
  document.documentElement.dataset.appearance = "light";
  document.documentElement.dataset.mode = mode;
  document.documentElement.style.colorScheme = mode;

  if (state.systemMq) {
    try {
      state.systemMq.onchange = null;
    } catch {}
    state.systemMq = null;
  }

  setActiveChoice(qsa(".appearanceOpt"), () => false);
}

function applyBackground(bg) {
  const next = (bg || "").trim() || "matching";
  setLsValue(LS.BG, next);
  document.documentElement.dataset.bg = next;
  setActiveChoice(qsa(".bgOpt"), (x) => (x.dataset.bg || "") === next);
}

function applyTheme(t) {
  const theme = (t || "").trim() || DEFAULTS.THEME;
  document.documentElement.dataset.theme = theme;
  setLsValue(LS.THEME, theme);
  if (ui.themeLabel) ui.themeLabel.textContent = theme[0].toUpperCase() + theme.slice(1);
  setActiveChoice(qsa(".themeOpt"), (b) => (b.dataset.theme || "") === theme);
}

function syncToolCheckboxesFromState() {
  const sel = new Set(getSelectedTools());
  if (ui.toolDb) ui.toolDb.checked = sel.has("database");
  if (ui.toolPlaceholder1) ui.toolPlaceholder1.checked = sel.has("placeholder1");
  if (ui.placeholderA) ui.placeholderA.checked = sel.has("placeholderA");
  if (ui.placeholderB) ui.placeholderB.checked = sel.has("placeholderB");
}

function readToolCheckboxesToState() {
  const next = [];
  if (ui.toolDb?.checked) next.push("database");
  if (ui.toolPlaceholder1?.checked) next.push("placeholder1");
  if (ui.placeholderA?.checked) next.push("placeholderA");
  if (ui.placeholderB?.checked) next.push("placeholderB");
  setSelectedTools(next);
  return next;
}

function openRightbar() {
  if (!ui.rightbar) return;
  ui.rightbar.classList.remove("hidden");
  ui.rightbar.classList.add("flex");
  setRightbarCollapsed(false);
  ui.rightbarBackdrop?.classList.remove("hidden");
  saveJsonLS(LS.RIGHTBAR, { open: true });
}

function closeRightbar() {
  if (!ui.rightbar) return;
  if (window.matchMedia?.("(min-width: 768px)").matches) {
    ui.rightbarBackdrop?.classList.add("hidden");
    saveJsonLS(LS.RIGHTBAR, { open: false });
    return;
  }
  ui.rightbar.classList.add("hidden");
  ui.rightbar.classList.remove("flex");
  ui.rightbarBackdrop?.classList.add("hidden");
  saveJsonLS(LS.RIGHTBAR, { open: false });
}

function setRightbarCollapsed(collapsed) {
  if (!ui.rightbar) return;

  if (collapsed) {
    ui.rightbar.classList.add("rightbar-collapsed", "is-collapsed");
    document.body.classList.add("rightbar-collapsed");

    ui.rightbar.dataset._prevWidth = ui.rightbar.style.width || "";
    ui.rightbar.dataset._prevMinWidth = ui.rightbar.style.minWidth || "";
    ui.rightbar.dataset._prevMaxWidth = ui.rightbar.style.maxWidth || "";
    ui.rightbar.dataset._prevFlex = ui.rightbar.style.flex || "";
    ui.rightbar.dataset._prevFlexBasis = ui.rightbar.style.flexBasis || "";
    ui.rightbar.dataset._prevPadding = ui.rightbar.style.padding || "";
    ui.rightbar.dataset._prevBorderWidth = ui.rightbar.style.borderWidth || "";
    ui.rightbar.dataset._prevOverflow = ui.rightbar.style.overflow || "";

    ui.rightbar.style.width = "0";
    ui.rightbar.style.minWidth = "0";
    ui.rightbar.style.maxWidth = "0";
    ui.rightbar.style.flex = "0 0 0";
    ui.rightbar.style.flexBasis = "0";
    ui.rightbar.style.padding = "0";
    ui.rightbar.style.borderWidth = "0";
    ui.rightbar.style.overflow = "hidden";

    const toggleBtn = $("rightbarToggleDesktop");
    if (toggleBtn) {
      toggleBtn.style.visibility = "visible";
      toggleBtn.style.display = "inline-flex";
    }
    return;
  }

  ui.rightbar.classList.remove("rightbar-collapsed", "is-collapsed");
  document.body.classList.remove("rightbar-collapsed");

  ui.rightbar.style.width = ui.rightbar.dataset._prevWidth || "";
  ui.rightbar.style.minWidth = ui.rightbar.dataset._prevMinWidth || "";
  ui.rightbar.style.maxWidth = ui.rightbar.dataset._prevMaxWidth || "";
  ui.rightbar.style.flex = ui.rightbar.dataset._prevFlex || "";
  ui.rightbar.style.flexBasis = ui.rightbar.dataset._prevFlexBasis || "";
  ui.rightbar.style.padding = ui.rightbar.dataset._prevPadding || "";
  ui.rightbar.style.borderWidth = ui.rightbar.dataset._prevBorderWidth || "";
  ui.rightbar.style.overflow = ui.rightbar.dataset._prevOverflow || "";

  delete ui.rightbar.dataset._prevWidth;
  delete ui.rightbar.dataset._prevMinWidth;
  delete ui.rightbar.dataset._prevMaxWidth;
  delete ui.rightbar.dataset._prevFlex;
  delete ui.rightbar.dataset._prevFlexBasis;
  delete ui.rightbar.dataset._prevPadding;
  delete ui.rightbar.dataset._prevBorderWidth;
  delete ui.rightbar.dataset._prevOverflow;
}

function initRightbarControls() {
  closeRightbar();

  const st = loadJsonLS(LS.RIGHTBAR, { open: false });
  if (st?.open) openRightbar();
  if (window.matchMedia?.("(min-width: 768px)").matches) setRightbarCollapsed(!st?.open);

  syncToolCheckboxesFromState();

  on(ui.rightbarToggle, "click", (e) => {
    e.preventDefault();
    if (ui.rightbar?.classList.contains("hidden")) openRightbar();
    else closeRightbar();
  });
  on(ui.rightbarClose, "click", (e) => {
    e.preventDefault();
    closeRightbar();
  });
  on(ui.rightbarBackdrop, "click", () => closeRightbar());

  on(ui.rightbarToggleDesktop, "click", (e) => {
    e.preventDefault();
    const collapsed = ui.rightbar?.classList.contains("rightbar-collapsed");
    if (collapsed) {
      setRightbarCollapsed(false);
      saveJsonLS(LS.RIGHTBAR, { open: true });
      return;
    }
    setRightbarCollapsed(true);
    ui.rightbarBackdrop?.classList.add("hidden");
    saveJsonLS(LS.RIGHTBAR, { open: false });
  });

  [ui.toolDb, ui.toolPlaceholder1, ui.placeholderA, ui.placeholderB].forEach((x) => {
    on(x, "change", () => readToolCheckboxesToState());
  });
}

function openSidebar() {
  if (!ui.sidebar) return;
  ui.sidebar.classList.remove("hidden");
  ui.sidebar.classList.add("flex");
  ui.sidebarBackdrop?.classList.remove("hidden");
}

function closeSidebar() {
  if (!ui.sidebar) return;
  if (window.matchMedia?.("(min-width: 768px)").matches) return;
  ui.sidebar.classList.add("hidden");
  ui.sidebar.classList.remove("flex");
  ui.sidebarBackdrop?.classList.add("hidden");
}

function initSidebarControls() {
  on(ui.sidebarToggle, "click", (e) => {
    e.preventDefault();
    openSidebar();
  });
  on(ui.sidebarClose, "click", (e) => {
    e.preventDefault();
    closeSidebar();
  });
  on(ui.sidebarBackdrop, "click", () => closeSidebar());

  on(ui.sidebarToggleDesktop, "click", (e) => {
    e.preventDefault();
    ui.sidebar?.classList.toggle("sidebar-collapsed");
    document.body.classList.toggle("sidebar-collapsed", ui.sidebar?.classList.contains("sidebar-collapsed"));
  });

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t && (t.closest?.(".chatMenu") || t.closest?.(".chatMenuSub") || t.closest?.(".chatMenuBtn"))) return;
    qsa(".chatMenu").forEach((m) => m.classList.add("hidden"));
    qsa(".chatMenuSub").forEach((m) => m.classList.add("hidden"));
  });
}

function openSettingsModal() {
  if (!ui.settingsModal) return;
  ui.settingsModal.classList.remove("hidden");
  ui.settingsModal.setAttribute("aria-hidden", "false");

  const panel = qs('[role="dialog"]', ui.settingsModal) || ui.settingsModal.firstElementChild;
  if (panel) {
    panel.style.left = "50%";
    panel.style.top = "50%";
    panel.style.transform = "translate(-50%, -50%)";
  }

  populateModelSelect();
}

function closeSettingsModal() {
  if (!ui.settingsModal) return;
  ui.settingsModal.classList.add("hidden");
  ui.settingsModal.setAttribute("aria-hidden", "true");

  const panel = qs('[role="dialog"]', ui.settingsModal) || ui.settingsModal.firstElementChild;
  if (panel) {
    panel.style.left = "";
    panel.style.top = "";
    panel.style.transform = "";
  }
}

function normalizeHexColor(value, fallback) {
  const raw = String(value || "").trim();
  const withHash = raw.startsWith("#") ? raw : `#${raw}`;
  if (/^#[0-9a-fA-F]{6}$/.test(withHash)) return withHash.toLowerCase();
  if (/^#[0-9a-fA-F]{3}$/.test(withHash)) {
    return `#${withHash[1]}${withHash[1]}${withHash[2]}${withHash[2]}${withHash[3]}${withHash[3]}`.toLowerCase();
  }
  return fallback;
}

function applyCustomColors(accent, toggleBg, chatBg) {
  const nextAccent = normalizeHexColor(accent, DEFAULTS.ACCENT);
  const nextToggleBg = normalizeHexColor(toggleBg, DEFAULTS.TOGGLE_BG);
  const nextChatBg = normalizeHexColor(chatBg, DEFAULTS.CHAT_BG);

  if (ui.accentHex) ui.accentHex.value = nextAccent;
  if (ui.accentPicker) ui.accentPicker.value = nextAccent;
  if (ui.toggleBgHex) ui.toggleBgHex.value = nextToggleBg;
  if (ui.toggleBgPicker) ui.toggleBgPicker.value = nextToggleBg;
  if (ui.chatBgHex) ui.chatBgHex.value = nextChatBg;
  if (ui.chatBgPicker) ui.chatBgPicker.value = nextChatBg;

  setLsValue(LS.ACCENT, nextAccent);
  setLsValue(LS.TOGGLE_BG, nextToggleBg);
  setLsValue(LS.CHAT_BG, nextChatBg);

  document.documentElement.style.setProperty("--accent", nextAccent);
  document.documentElement.style.setProperty("--toggle-bg", nextToggleBg);
  document.documentElement.style.setProperty("--chat-bg", nextChatBg);
}

async function fetchModels() {
  return await fetchJson("/app/api/models");
}

async function populateModelSelect() {
  if (!ui.modelSelect) return;

  let data;
  try {
    data = await fetchModels();
  } catch {
    return;
  }

  const models = Array.isArray(data?.models)
    ? data.models.filter((m) => typeof m === "string" && m.trim())
    : [];
  if (!models.length) return;

  const saved = getLsValue(LS.MODEL, "");
  const preferred = (saved || data?.default || "").trim();

  ui.modelSelect.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    ui.modelSelect.appendChild(opt);
  });

  ui.modelSelect.value = preferred && models.includes(preferred) ? preferred : models[0];
  setLsValue(LS.MODEL, ui.modelSelect.value);
}

function initSettingsModal() {
  applyTheme(getLsValue(LS.THEME, DEFAULTS.THEME));
  applyAppearance("light");
  applyBackground(getLsValue(LS.BG, DEFAULTS.BG));

  applyCustomColors(
    getLsValue(LS.ACCENT, DEFAULTS.ACCENT),
    getLsValue(LS.TOGGLE_BG, DEFAULTS.TOGGLE_BG),
    getLsValue(LS.CHAT_BG, DEFAULTS.CHAT_BG)
  );

  on(ui.settingsModalBtn, "click", (e) => {
    e.preventDefault();
    openSettingsModal();
  });

  qsa('[data-close="settings"]', ui.settingsModal || document).forEach((x) => {
    x.addEventListener("click", (e) => {
      e.preventDefault();
      closeSettingsModal();
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSettingsModal();
  });

  on(ui.accentPicker, "input", () => {
    applyCustomColors(
      ui.accentPicker.value,
      ui.toggleBgHex?.value || ui.toggleBgPicker?.value || DEFAULTS.TOGGLE_BG,
      ui.chatBgHex?.value || ui.chatBgPicker?.value || DEFAULTS.CHAT_BG
    );
  });

  on(ui.accentHex, "change", () => {
    applyCustomColors(
      ui.accentHex.value,
      ui.toggleBgHex?.value || ui.toggleBgPicker?.value || DEFAULTS.TOGGLE_BG,
      ui.chatBgHex?.value || ui.chatBgPicker?.value || DEFAULTS.CHAT_BG
    );
  });

  const syncToggleBg = (value) => {
    applyCustomColors(
      ui.accentHex?.value || ui.accentPicker?.value || DEFAULTS.ACCENT,
      value,
      ui.chatBgHex?.value || ui.chatBgPicker?.value || DEFAULTS.CHAT_BG
    );

    const cur = document.documentElement.dataset.bg || DEFAULTS.BG;
    if (cur !== "gray") {
      applyBackground("gray");
      setLsValue(LS.BG, "gray");
      if (ui.themeLabel) ui.themeLabel.textContent = "Gray";
    }
  };

  on(ui.toggleBgPicker, "input", () => syncToggleBg(ui.toggleBgPicker.value));
  on(ui.toggleBgHex, "change", () => syncToggleBg(ui.toggleBgHex.value));

  on(ui.chatBgPicker, "input", () => {
    applyCustomColors(
      ui.accentHex?.value || ui.accentPicker?.value || DEFAULTS.ACCENT,
      ui.toggleBgHex?.value || ui.toggleBgPicker?.value || DEFAULTS.TOGGLE_BG,
      ui.chatBgPicker.value
    );
  });

  on(ui.chatBgHex, "change", () => {
    applyCustomColors(
      ui.accentHex?.value || ui.accentPicker?.value || DEFAULTS.ACCENT,
      ui.toggleBgHex?.value || ui.toggleBgPicker?.value || DEFAULTS.TOGGLE_BG,
      ui.chatBgHex.value
    );
  });

  on(ui.restoreDefaultsBtn, "click", (e) => {
    e.preventDefault();
    applyCustomColors(DEFAULTS.ACCENT, DEFAULTS.TOGGLE_BG, DEFAULTS.CHAT_BG);
  });

  document.addEventListener("change", (e) => {
    if (e.target?.matches('#rightbar input[type="date"]')) syncDateInputState();
  });
  document.addEventListener("input", (e) => {
    if (e.target?.matches('#rightbar input[type="date"]')) syncDateInputState();
  });

  const savedModel = getLsValue(LS.MODEL, "");
  if (ui.modelSelect && savedModel) ui.modelSelect.value = savedModel;
  on(ui.modelSelect, "change", () => setLsValue(LS.MODEL, ui.modelSelect.value));

  on(ui.scheduleTaskBtn, "click", (e) => {
    e.preventDefault();
    alert("Task scheduling UI coming next.");
  });
}

function bindChatControls() {
  on(ui.newChatBtn, "click", () => {
    state.activeSessionId = newSessionId();
    state.activeMessages = [];
    state.pendingAttachments = [];
    renderAttachmentStrip();
    setHeaderTitle();
    renderChatList(state.sessions);
    renderMessages(state.activeMessages);
    ui.input?.focus();
  });

  on(ui.newProjectBtn, "click", (e) => {
    e.preventDefault();
    createProject();
  });

  on(ui.chatSearch, "input", () => renderChatList(state.sessions));
  on(ui.sendBtn, "click", sendCurrent);

  on(ui.attachBtn, "click", (e) => {
    e.preventDefault();
    ui.fileInput?.click();
  });

  on(ui.fileInput, "change", async () => {
    try {
      await uploadSelectedFiles(ui.fileInput.files);
    } catch (err) {
      alert("Upload failed: " + (err?.message || err));
    } finally {
      if (ui.fileInput) ui.fileInput.value = "";
    }
  });

  on(ui.input, "input", autosizeTextarea);
  on(ui.input, "keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendCurrent();
    }
  });
}

async function init() {
  autosizeTextarea();
  initSidebarControls();
  initRightbarControls();
  initSettingsModal();
  bindChatControls();
  syncDateInputState();
  renderProjects();
  await loadSessions();
}

init();