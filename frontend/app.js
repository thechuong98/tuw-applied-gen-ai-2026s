// Behind the nginx proxy, /api is same-origin. For standalone dev set window.API_BASE = "http://localhost:8000".
const API_BASE = window.API_BASE || "";

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");
const resultBox = $("result");
const statusEl = $("status");
const runBtn = $("run");

// Prefill with the moon-landing example from the topic description.
$("text").value =
  "I remember watching the moon landing with my father. It was a huge event to see Neil Armstrong " +
  "become the first man on the Moon. Funnily enough, this is the only specific memory I have from " +
  "when I was six years old.";
$("attrs").value = "age, year_of_birth";

const ICON = { start: "▶️", defender: "🛡️", attacker: "🕵️", judge: "⚖️" };
const TITLE = { defender: "Defender", attacker: "Attacker", judge: "Judge" };

let lastRound = 0;

function clearUI() {
  timeline.innerHTML = "";
  resultBox.className = "result hidden";
  resultBox.innerHTML = "";
  lastRound = 0;
}

function roundSep(n) {
  if (n === lastRound) return;
  lastRound = n;
  const el = document.createElement("div");
  el.className = "round-sep";
  el.textContent = `— Round ${n} —`;
  timeline.appendChild(el);
}

function card(kind, headHtml, bodyHtml) {
  const el = document.createElement("div");
  el.className = `card ${kind}`;
  el.innerHTML = `<div class="card-head">${ICON[kind] || ""} ${headHtml}</div><div class="card-body">${bodyHtml}</div>`;
  timeline.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function render(ev) {
  if (ev.type === "start") {
    card("start", "Start", `Hiding: <b>${esc((ev.attributes_to_hide || []).join(", "))}</b>`);
    return;
  }
  if (ev.type === "error") {
    card("error", "Error", esc(ev.message));
    return;
  }
  if (ev.type === "done") {
    renderResult(ev);
    return;
  }
  // node events
  roundSep(ev.round);
  if (ev.node === "defender") {
    const reason = ev.reasoning ? `<div class="g-reason"><i>reasoning:</i> ${esc(ev.reasoning)}</div>` : "";
    const chips = Object.entries(ev.strategy_log || {})
      .map(([k, v]) => `<span class="chip"><b>${esc(k)}</b>: ${esc(v)}</span>`).join("");
    card("defender", `${TITLE.defender} rewrites`,
      `<div class="text-block">${esc(ev.rewritten_text)}</div>${reason}${chips ? `<div class="chips">${chips}</div>` : ""}`);
  } else if (ev.node === "attacker") {
    const rows = (ev.guesses || []).map((g) => {
      const val = g.guess == null ? '<span class="g-meta">unknown</span>' : `<span class="g-val">${esc(g.guess)}</span>`;
      const ev2 = (g.evidence_spans || []).length ? `<div class="g-meta">clue: ${esc(g.evidence_spans.join(" · "))}</div>` : "";
      return `<div class="guess"><span class="g-attr">${esc(g.attribute)}</span> → ${val}
        <span class="g-meta">(conf ${(g.confidence || 0).toFixed(2)})</span>
        ${g.reasoning ? `<div class="g-reason">${esc(g.reasoning)}</div>` : ""}${ev2}</div>`;
    }).join("");
    card("attacker", `${TITLE.attacker} reasons`, rows || "<i>no guesses</i>");
  } else if (ev.node === "judge") {
    const badge = ev.leaked
      ? `<span class="badge bad">LEAKED: ${esc((ev.leaked_attrs || []).join(", "))}</span>`
      : `<span class="badge ok">NO LEAK</span>`;
    const leakRows = (ev.leaks || []).filter((l) => l.leaked).map((l) =>
      `<div class="g-reason"><b>${esc(l.attribute)}</b>${l.inferred_value ? ` → ${esc(l.inferred_value)}` : ""}: ${esc(l.rationale || "")}</div>`).join("");
    const summaryRow = ev.summary ? `<div class="g-reason"><i>why ${ev.leaked ? "leaked" : "safe"}:</i> ${esc(ev.summary)}</div>` : "";
    const s = ev.scores || {};
    const fields = ["task_utility", "informational_completeness", "factual_consistency", "fluency", "format_preserved"];
    const bars = fields.filter((f) => f in s).map((f) => {
      const pct = Math.round((s[f] || 0) * 100);
      return `<div>${f}</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><div>${(s[f] || 0).toFixed(2)}</div>`;
    }).join("");
    const notesRow = s.notes ? `<div class="g-reason"><i>utility reason:</i> ${esc(s.notes)}</div>` : "";
    card("judge", TITLE.judge,
      `<div class="verdict-line">Privacy: ${badge} ${ev.leaked ? "→ Defender rewrites harder" : ""}</div>${leakRows}${summaryRow}<div class="bars">${bars}</div>${notesRow}`);
  }
}

function renderResult(ev) {
  const pass = ev.verdict === "PASS";
  const badge = pass ? '<span class="badge ok">PASS</span>' : '<span class="badge warn">MAX_ITERS (best effort)</span>';
  resultBox.className = "result";
  resultBox.innerHTML = `
    <button class="copy" id="copy">Copy</button>
    <h3>Result ${badge}</h3>
    <div class="g-meta">After ${ev.rounds} round(s)${pass ? " · Judge passed: no leak & utility preserved" : " · returning best candidate"}</div>
    <div class="text-block" id="final">${esc(ev.final_text)}</div>`;
  $("copy").onclick = () => navigator.clipboard.writeText(ev.final_text || "");
}

async function run() {
  const text = $("text").value.trim();
  if (!text) { statusEl.textContent = "Enter some text first."; return; }
  const attrs = $("attrs").value.split(",").map((s) => s.trim()).filter(Boolean);
  const utility = $("utility").value.split(",").map((s) => s.trim()).filter(Boolean);

  clearUI();
  runBtn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span> Running the adversarial loop...';

  try {
    const resp = await fetch(`${API_BASE}/api/anonymize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, attributes_to_hide: attrs, utility_to_preserve: utility }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (line.trim()) render(JSON.parse(line));
      }
    }
    statusEl.textContent = "Done.";
  } catch (e) {
    card("error", "Connection error", esc(e.message));
    statusEl.textContent = "Failed.";
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", run);
