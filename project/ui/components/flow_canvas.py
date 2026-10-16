"""Process-flow canvas: flowchart symbols on a pannable grid, a detail drawer per node, and playback.

Everything interactive runs in the browser; Python only passes the flow from flow_model.flow().
"""
import streamlit as st

_CSS = """
.ato-flow { position: relative; height: 720px; border: 1px solid #d6d6d6; border-radius: 10px; overflow: hidden;
  background-color: #f4f4f4; font-family: "Source Sans 3", "Source Sans Pro", system-ui, sans-serif; color: #1f2328;
  user-select: none; }
.ato-flow.grid { background-image: linear-gradient(#e3e3e3 1px, transparent 1px), linear-gradient(90deg, #e3e3e3 1px, transparent 1px); }
.ato-flow.dots { background-image: radial-gradient(#c8c8c8 1.2px, transparent 1.2px); }
.ato-flow svg { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }
.ato-flow svg.dragging { cursor: grabbing; }
.ato-bar { position: absolute; top: 10px; left: 10px; display: flex; gap: 6px; align-items: center; z-index: 3; flex-wrap: wrap; }
.ato-bar button { border: 1px solid #c9c9c9; background: #fff; border-radius: 6px; padding: 4px 10px; font: inherit;
  font-size: 13px; cursor: pointer; color: #1f2328; }
.ato-bar button:hover { background: #f0f0f0; }
.ato-bar button.on { background: #1f2328; color: #fff; border-color: #1f2328; }
.ato-tag { font-size: 12px; padding: 3px 8px; border-radius: 999px; background: #fff3cd; border: 1px solid #e8c766; }
.ato-step { font-size: 13px; color: #555; background: #ffffffcc; padding: 3px 8px; border-radius: 6px; }
.node text { font-size: 13px; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
.node .sub { font-size: 10.5px; fill: #6b6b6b; }
.node .shape { fill: #fff; stroke: #2b2b2b; stroke-width: 1.2; }
.node.model .shape { stroke: #6d4fc2; }
.node.engineer .shape { stroke: #b7791f; }
.node.decision .shape { fill: #a7f0cf; }
.node.failed .shape { stroke: #c62828; stroke-width: 2; fill: #fdecea; }
.node.pending .shape { stroke-dasharray: 5 4; fill: #fafafa; }
.node { cursor: pointer; transition: opacity .25s; }
.node:hover .shape { stroke-width: 2.2; }
.node.selected .shape { stroke: #0b6bcb; stroke-width: 2.6; }
.node.current .shape { filter: drop-shadow(0 0 6px #0b6bcb); stroke: #0b6bcb; stroke-width: 2.6; }
.node.hidden, .edge.hidden { opacity: 0.08; }
.edge path { fill: none; stroke: #2b2b2b; stroke-width: 2.4; }
.edge.dashed path { stroke-dasharray: 7 5; stroke: #777; }
.edge text { font-size: 12px; font-weight: 600; fill: #2b2b2b; }
.edge { transition: opacity .25s; }
.ato-legend { position: absolute; bottom: 10px; left: 10px; display: flex; gap: 12px; font-size: 12px; background: #ffffffd9;
  padding: 5px 9px; border-radius: 6px; z-index: 3; }
.ato-legend i { display: inline-block; width: 10px; height: 10px; border: 2px solid; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.ato-drawer { position: absolute; top: 0; right: 0; width: 440px; max-width: 70%; height: 100%; background: #fff;
  border-left: 1px solid #d6d6d6; box-shadow: -4px 0 14px #0000001a; z-index: 4; overflow-y: auto; padding: 14px 16px;
  transform: translateX(105%); transition: transform .22s; user-select: text; }
.ato-drawer.open { transform: none; }
.ato-drawer h3 { margin: 0 0 2px; font-size: 17px; }
.ato-drawer .meta { font-size: 12px; color: #666; margin-bottom: 10px; }
.ato-drawer .close { position: absolute; top: 10px; right: 12px; border: none; background: none; font-size: 18px; cursor: pointer; }
.ato-drawer details { margin: 8px 0; border: 1px solid #e3e3e3; border-radius: 6px; padding: 6px 8px; }
.ato-drawer summary { cursor: pointer; font-weight: 600; font-size: 13px; }
.ato-drawer pre { white-space: pre-wrap; word-break: break-word; font-size: 11.5px; background: #f7f7f7; padding: 8px;
  border-radius: 4px; max-height: 340px; overflow: auto; margin: 6px 0 0; }
.ato-drawer table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 4px; }
.ato-drawer td { border-top: 1px solid #eee; padding: 5px 4px; vertical-align: top; }
.ato-drawer .k { font-family: ui-monospace, monospace; font-size: 11px; color: #444; }
.chip { display: inline-block; font-size: 10.5px; padding: 1px 6px; border-radius: 999px; margin: 1px 2px 1px 0; border: 1px solid; }
.chip.kept, .chip.shown, .chip.applied, .chip.created { color: #146c2e; border-color: #8fd19e; background: #eefaf0; }
.chip.dropped, .chip.capped, .chip.rejected, .chip.hidden { color: #9a3412; border-color: #f2b38a; background: #fff4ec; }
.chip.relabeled, .chip.merged { color: #6d4fc2; border-color: #c4b5f0; background: #f4f0ff; }
.chip.code { color: #333; border-color: #bbb; }
.chip.model { color: #6d4fc2; border-color: #c4b5f0; }
.chip.engineer { color: #b7791f; border-color: #ecc98b; }
.chip.decision, .chip.data, .chip.unsupported { color: #0b4f8a; border-color: #9cc3e8; background: #eef5fc; }
.rule { font-size: 11px; color: #666; }
.cite { border: 1px solid #c9d3df; background: #fff; color: #0b4f8a; border-radius: 4px; font: 11px ui-monospace, monospace;
  padding: 0 5px; margin-left: 4px; cursor: pointer; white-space: nowrap; }
.cite:hover { background: #eef5fc; }
.codebox { margin: 6px 0 8px; border: 1px solid #d6dee8; border-radius: 6px; overflow: hidden; background: #fbfcfe; }
.codebox[hidden] { display: none; }
.codehead { display: flex; justify-content: space-between; gap: 8px; font: 11px ui-monospace, monospace; background: #eef2f7;
  padding: 4px 8px; color: #334; }
.codehead a { color: #0b6bcb; text-decoration: none; font-family: "Source Sans 3", system-ui, sans-serif; }
.codebox pre.code { margin: 0; max-height: 300px; overflow: auto; background: none; padding: 6px 0; font-size: 11px; }
.code .ln { display: inline-block; width: 38px; color: #98a2b0; text-align: right; padding-right: 8px; user-select: none; }
.code .row { display: block; padding-right: 8px; white-space: pre; }
.code .hl { background: #fff3c4; }
.about { background: #f6f8fb; border: 1px solid #e3e8ef; border-radius: 6px; padding: 6px 10px; margin: 4px 0 10px; font-size: 13px; }
.about p { margin: 2px 0 4px; font-weight: 600; }
.about ul { margin: 0 0 2px 16px; padding: 0; }
.about li { margin: 2px 0; }
.pill rect { fill: #f4f0ff; stroke: #6d4fc2; stroke-width: 1; }
.pill text { font-size: 10.5px; fill: #4b2fa0; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
.pill { cursor: help; }
.pill:hover rect { fill: #e6ddff; }
.ato-pop { position: absolute; z-index: 5; width: 460px; max-height: 62%; overflow-y: auto; background: #fff;
  border: 1px solid #c4b5f0; border-radius: 8px; box-shadow: 0 6px 22px #00000026; padding: 10px 14px; display: none;
  font-size: 12.5px; line-height: 1.45; user-select: text; }
.ato-pop h4, .prompt h4 { margin: 8px 0 4px; font-size: 12.5px; color: #4b2fa0; }
.prompt-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
.prompt p { margin: 4px 0; }
.prompt ul { margin: 2px 0 6px 16px; padding: 0; }
.prompt li { margin: 2px 0; }
.prompt code { background: #f4f0ff; padding: 0 3px; border-radius: 3px; font-size: 11.5px; }
"""

_JS = r"""
const NS = "http://www.w3.org/2000/svg";
const CW = 190, RH = 150, W = 150, H = 62, DW = 124, DH = 92;
const views = {};

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}
function box(n) {
  const x = 80 + n.col * CW, y = 90 + n.row * RH;
  const w = n.shape === "decision" ? DW : W, h = n.shape === "decision" ? DH : H;
  return {x, y, w, h, l: x - w / 2, r: x + w / 2, t: y - h / 2, b: y + h / 2};
}
function shapePath(n, b) {
  const {l, r, t, w, h} = b, bt = b.b;
  switch (n.shape) {
    case "decision": return `M${b.x},${t} L${r},${b.y} L${b.x},${bt} L${l},${b.y} Z`;
    case "data": return `M${l + 16},${t} L${r},${t} L${r - 16},${bt} L${l},${bt} Z`;
    case "document": return `M${l},${t} L${r - 14},${t} L${r},${t + 14} L${r},${bt - 6} ` +
      `Q${r - w / 4},${bt + 8} ${b.x},${bt - 4} T${l},${bt - 4} Z M${r - 14},${t} L${r - 14},${t + 14} L${r},${t + 14}`;
    case "folder": return `M${l},${t + 9} L${r - 46},${t + 9} L${r - 38},${t} L${r},${t} L${r},${bt} L${l},${bt} Z`;
    default: return `M${l},${t} L${r},${t} L${r},${bt} L${l},${bt} Z`;
  }
}
function route(a, b, exit) {
  if (exit === "left") return [[a.l, a.y], [b.x, a.y], [b.x, b.t]];   // along the clear row, then down
  if (Math.abs(a.y - b.y) < 2) {
    return b.x > a.x ? [[a.r, a.y], [b.l, b.y]] : [[a.l, a.y], [b.r, b.y]];
  }
  if (Math.abs(a.x - b.x) < 2) {
    return b.y > a.y ? [[a.x, a.b], [b.x, b.t]] : [[a.x, a.t], [b.x, b.b]];
  }
  if (b.y > a.y) {
    if (b.x > a.x) return [[a.r, a.y], [(a.r + b.l) / 2, a.y], [(a.r + b.l) / 2, b.y], [b.l, b.y]];
    const midY = (a.b + b.t) / 2;
    return [[a.x, a.b], [a.x, midY], [b.x, midY], [b.x, b.t]];
  }
  if (b.x > a.x) return [[a.r, a.y], [(a.r + b.l) / 2, a.y], [(a.r + b.l) / 2, b.y], [b.l, b.y]];
  const midY = (a.t + b.b) / 2;
  return [[a.x, a.t], [a.x, midY], [b.x, midY], [b.x, b.b]];
}
function table(rows) {
  return `<table>${rows.join("")}</table>`;
}
function chip(cls, text) { return text ? `<span class="chip ${esc(cls)}">${esc(text)}</span>` : ""; }
function itemsTable(items) {
  if (!items || !items.length) return "<p class='rule'>Nothing recorded.</p>";
  return table(items.map(i => {
    const d = i.detail || {};
    const extra = [d.via_example ? "used the example answer" : "", d.other_choice ? "typed an Other answer" : "",
                   d.question ? `Q: ${d.question}` : "", d.old !== undefined && d.old !== null ? `old: ${JSON.stringify(d.old)}` : "",
                   d.new !== undefined && d.new !== null ? `new: ${JSON.stringify(d.new)}` : ""].filter(Boolean).join(" · ");
    return `<tr><td style="width:36%"><div class="k">${esc(i.key)}</div>${chip(i.type, i.type)}${chip(i.origin, i.origin)}${chip(i.kind, i.kind)}</td>
      <td>${esc(i.text)}${i.why ? `<div class="rule">Why: ${esc(i.why)}</div>` : ""}
      <div>${chip(i.fate, i.fate)}${i.fate_rule || i.rule ? `<span class="k">${esc(i.fate_rule || i.rule)}</span>` : ""}${citeHtml(i.rule_cite)}</div>
      ${i.rule_text ? `<div class="rule">${esc(i.rule_text)}</div>` : ""}${extra ? `<div class="rule">${esc(extra)}</div>` : ""}</td></tr>`;
  }));
}
function inline(text) {
  return esc(text).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/&quot;([^&]{1,40}?)&quot;/g, "<code>&quot;$1&quot;</code>");
}
function prettyPrompt(text) {
  const lines = String(text || "").split("\n");
  let html = "", para = [], items = [], inList = false;
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const flushList = () => { if (items.length) { html += "<ul>" + items.map(i => `<li>${inline(i)}</li>`).join("") + "</ul>"; items = []; } inList = false; };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if (line.startsWith("- ")) { flushPara(); inList = true; items.push(line.slice(2)); continue; }
    if (inList && /^\s+/.test(line)) { items[items.length - 1] += " " + line.trim(); continue; }
    flushList();
    if (/^[A-Z][^.]{0,40}:$/.test(line.trim())) { flushPara(); html += `<h4>${esc(line.trim().slice(0, -1))}</h4>`; continue; }
    para.push(line.trim());
  }
  flushPara(); flushList();
  return `<div class="prompt">${html}</div>`;
}
function citeHtml(c) {
  // A clickable file:line chip; the code it points to opens inline below it.
  if (!c) return "";
  const rows = c.lines.map((line, k) => {
    const n = c.first + k, hl = n >= c.start && n <= c.end && c.end - c.start < 40;
    return `<span class="row${hl ? " hl" : ""}"><span class="ln">${n}</span>${esc(line)}</span>`;
  }).join("");
  return `<button class="cite" type="button" title="Show the code behind this">‹/› ${esc(c.label)}</button>` +
    `<div class="codebox" hidden><div class="codehead"><span>${esc(c.file)} · lines ${c.start}–${c.end}${c.truncated ? ` (first ${c.shown} lines shown)` : ""}</span>` +
    (c.open ? `<a href="${esc(c.open)}" title="Open this line in VS Code on your machine">Open in VS Code ↗</a>` : "") +
    `</div><pre class="code">${rows}</pre></div>`;
}
function promptsHtml(d) {
  if (!d.prompts || !d.prompts.length) return "";
  return `<details open><summary>Prompts sent (${d.prompts.length} parts)</summary>` +
    (d.prompt_note ? `<p class="rule">${esc(d.prompt_note)}</p>` : "") +
    d.prompts.map(p => `<details><summary>${esc(p.name)}</summary>${prettyPrompt(p.text)}</details>`).join("") + "</details>";
}
function block(title, text, open) {
  return text ? `<details ${open ? "open" : ""}><summary>${esc(title)}</summary><pre>${esc(text)}</pre></details>` : "";
}
function detailHtml(n) {
  const d = n.detail || {};
  let html = `<button class="close" title="Close">×</button><h3>${esc(n.label)}</h3>
    <div class="meta">${esc(n.shape)} · ${esc(n.actor)}${n.sub ? " · " + esc(n.sub) : ""}${n.status !== "ok" ? " · " + esc(n.status) : ""}</div>`;
  if (d.about) html += `<div class="about"><p>${esc(d.about.text)}${citeHtml(d.about.cite)}</p><ul>` +
    (d.about.points || []).map(x => `<li>${esc(x.text)}${citeHtml(x.cite)}</li>`).join("") + `</ul></div>`;
  if (d.text) html += `<p>${esc(d.text)}</p>`;
  if (d.metrics && Object.keys(d.metrics).length) html += table(Object.entries(d.metrics).map(([k, v]) =>
    `<tr><td class="k">${esc(k)}</td><td>${esc(typeof v === "number" ? (Math.abs(v) > 1e4 ? v.toExponential(3) : +v.toFixed(4)) : v)}</td></tr>`));
  if (d.image) html += `<img src="${d.image}" style="width:100%;margin-top:8px;border:1px solid #eee;border-radius:6px" alt="Convergence history">`;
  if (d.history_tail) html += block("Last iterations", d.history_tail.map(r => `${r.iteration}: volume ${r.volume_fraction.toFixed(4)}, compliance ${r.compliance.toPrecision(5)}, p-norm ${(r.stress_pnorm_pa / 1e6).toFixed(2)} MPa`).join("\n"), true);
  if (d.summary_notes) html += `<ul class="rule">${d.summary_notes.map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;
  if (d.context) html += block("Context", d.context, true);
  if (d.model !== undefined) {
    html += `<p class="rule">Model: ${esc(d.model)}${d.cache_hit ? " (cached reply)" : ""} · tokens in/out: ${esc((d.tokens || {}).in)} / ${esc((d.tokens || {}).out)} · ${esc(d.latency_ms)} ms</p>`;
    html += promptsHtml(d);
    html += block("Input sent (JSON)", d.payload, false);
    html += block("Model reply (raw)", d.reply, true);
  }
  if (d.reply && d.model === undefined) html += block("Model reply (raw)", d.reply, true);
  if (d.items) html += `<details open><summary>Items and what became of them (${d.items.length})</summary>${itemsTable(d.items)}</details>`;
  if (d.claimed_resolved) html += `<p class="rule">Model says it resolved: ${esc(d.claimed_resolved.join(", "))}</p>`;
  if (d.resolved_check) html += `<p class="rule">Claimed resolved but still open: ${esc((d.resolved_check.still_open || []).join(", ") || "none")}</p>`;
  if (d.config_built !== undefined) html += `<p class="rule">Solver configuration built: ${d.config_built ? "yes" : "no"}</p>`;
  if (d.spec) html += block("Specification (JSON)", d.spec, true);
  if (d.rejected) html += block("Rejected edits (JSON)", d.rejected, true);
  if (d.plan_hash) html += `<p class="rule">Plan hash: <span class="k">${esc(d.plan_hash)}</span></p>`;
  if (d.block_config) html += block("Block config", JSON.stringify(d.block_config, null, 2), true);
  if (d.solver_inputs) html += block("Solver inputs", JSON.stringify(d.solver_inputs, null, 2), false);
  if (d.budget) html += `<p class="rule">Budget: ${esc(JSON.stringify(d.budget))}</p>`;
  if (d.state) html += `<p class="rule">State after this step: ${esc(d.state)}</p>`;
  return html;
}

export default function (component) {
  const { data, parentElement } = component;
  const flow = data || {nodes: [], edges: []};
  const key = flow.key || "flow";
  const view = views[key] || (views[key] = {x: 20, y: 30, k: 0.78, bg: "grid", selected: null, step: -1, playing: false});
  parentElement.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = flow.css || "";
  parentElement.appendChild(style);
  const root = document.createElement("div");
  root.className = "ato-flow " + view.bg;
  parentElement.appendChild(root);

  const bar = document.createElement("div");
  bar.className = "ato-bar";
  bar.innerHTML = `${flow.origin === "simulation" ? '<span class="ato-tag">Simulated run · scripted model replies, real pipeline code</span>' : ""}
    <button data-a="play">▶ Play</button><button data-a="prev">◀</button><button data-a="next">▶|</button>
    <button data-a="all">Show all</button><button data-a="fit">Fit</button>
    <button data-a="grid" class="${view.bg === "grid" ? "on" : ""}">Grid</button><button data-a="dots" class="${view.bg === "dots" ? "on" : ""}">Dots</button>
    <span class="ato-step"></span>`;
  root.appendChild(bar);
  const legend = document.createElement("div");
  legend.className = "ato-legend";
  legend.innerHTML = '<span><i style="border-color:#6d4fc2"></i>model</span><span><i style="border-color:#2b2b2b"></i>code</span>' +
    '<span><i style="border-color:#b7791f"></i>engineer</span><span><i style="border-color:#c62828;background:#fdecea"></i>failed</span>' +
    '<span>▭ process ▱ data ⬭ review document ◇ decision 🗀 record</span>';
  root.appendChild(legend);
  const drawer = document.createElement("div");
  drawer.className = "ato-drawer";
  root.appendChild(drawer);
  const pop = document.createElement("div");
  pop.className = "ato-pop";
  root.appendChild(pop);
  let hideTimer = null;
  const hidePop = () => { hideTimer = setTimeout(() => { pop.style.display = "none"; }, 220); };
  pop.addEventListener("mouseenter", () => clearTimeout(hideTimer));
  pop.addEventListener("mouseleave", hidePop);
  pop.addEventListener("wheel", ev => ev.stopPropagation());
  function showPrompt(part, node, ev) {
    clearTimeout(hideTimer);
    pop.innerHTML = `<div class="prompt-title">${esc(node.label)} · ${esc(part.name)}</div>` +
      (node.detail.prompt_note ? `<p class="rule">${esc(node.detail.prompt_note)}</p>` : "") + prettyPrompt(part.text);
    const r = root.getBoundingClientRect();
    pop.style.display = "block";
    const x = Math.min(ev.clientX - r.left + 12, r.width - pop.offsetWidth - 10);
    const y = Math.min(ev.clientY - r.top + 12, r.height - pop.offsetHeight - 10);
    pop.style.left = Math.max(10, x) + "px";
    pop.style.top = Math.max(10, y) + "px";
  }

  const svg = el("svg", {});
  root.appendChild(svg);
  const defs = el("defs", {}, svg);
  const marker = el("marker", {id: "arrow-" + key, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse"}, defs);
  el("path", {d: "M0,0 L10,5 L0,10 z", fill: "#2b2b2b"}, marker);
  const g = el("g", {}, svg);

  const boxes = {}, nodeEls = {}, edgeEls = [];
  flow.nodes.forEach(n => { boxes[n.id] = box(n); });
  flow.edges.forEach(e => {
    const a = boxes[e.from], b = boxes[e.to];
    if (!a || !b) return;
    const pts = route(a, b, e.exit);
    const ge = el("g", {class: "edge" + (e.dashed ? " dashed" : "")}, g);
    el("path", {d: "M" + pts.map(p => p.join(",")).join(" L"), "marker-end": `url(#arrow-${key})`}, ge);
    if (e.label) {
      const [p0, p1] = pts;
      const horizontal = Math.abs(p0[1] - p1[1]) < 1;
      const leftward = horizontal && p1[0] < p0[0];
      const t = el("text", {x: p0[0] + (leftward ? -34 : horizontal ? 10 : 8), y: p0[1] + (horizontal ? -8 : 16)}, ge);
      t.textContent = e.label;
    }
    edgeEls.push({ge, e});
  });
  flow.nodes.forEach(n => {
    const b = boxes[n.id];
    const gn = el("g", {class: `node ${n.shape} ${n.actor} ${n.status}`}, g);
    el("path", {class: "shape", d: shapePath(n, b)}, gn);
    const label = el("text", {x: b.x, y: b.y - (n.sub ? 7 : 0)}, gn);
    label.textContent = n.label;
    if (n.sub) { const s = el("text", {class: "sub", x: b.x, y: b.y + 11}, gn); s.textContent = n.sub; }
    gn.addEventListener("click", ev => { ev.stopPropagation(); select(n.id); });
    const parts = (n.detail && n.detail.prompts) || [];
    if (parts.length) {
      // One chip per prompt part sent at this step; hover shows the part.
      const widths = parts.map(p => 18 + p.name.length * 5.6), gap = 6;
      let x = b.x - (widths.reduce((s, w) => s + w, 0) + gap * (parts.length - 1)) / 2;
      parts.forEach((part, i) => {
        const pg = el("g", {class: "pill"}, gn);
        el("rect", {x, y: b.b + 6, width: widths[i], height: 17, rx: 8.5}, pg);
        const pt = el("text", {x: x + widths[i] / 2, y: b.b + 15}, pg);
        pt.textContent = "✎ " + part.name;
        pg.addEventListener("mouseenter", ev => showPrompt(part, n, ev));
        pg.addEventListener("mouseleave", hidePop);
        pg.addEventListener("click", ev => { ev.stopPropagation(); select(n.id); });
        x += widths[i] + gap;
      });
    }
    nodeEls[n.id] = gn;
  });

  const order = [...flow.nodes].sort((a, b) => a.order - b.order).map(n => n.id);
  const stepLabel = bar.querySelector(".ato-step");
  function apply() {
    g.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`);
    root.style.backgroundSize = `${24 * view.k}px ${24 * view.k}px`;
    root.style.backgroundPosition = `${view.x}px ${view.y}px`;
    const revealed = view.step < 0 ? null : new Set(order.slice(0, view.step + 1));
    for (const id in nodeEls) {
      nodeEls[id].classList.toggle("hidden", !!revealed && !revealed.has(id));
      nodeEls[id].classList.toggle("selected", view.selected === id);
      nodeEls[id].classList.toggle("current", view.step >= 0 && order[view.step] === id);
    }
    edgeEls.forEach(({ge, e}) => ge.classList.toggle("hidden", !!revealed && !(revealed.has(e.from) && revealed.has(e.to))));
    stepLabel.textContent = view.step < 0 ? `${order.length} steps` : `Step ${view.step + 1} of ${order.length}`;
    bar.querySelector('[data-a="play"]').textContent = view.playing ? "❚❚ Pause" : "▶ Play";
  }
  function select(id) {
    view.selected = id;
    const n = flow.nodes.find(x => x.id === id);
    if (!n) { drawer.classList.remove("open"); apply(); return; }
    drawer.innerHTML = detailHtml(n);
    drawer.querySelectorAll(".cite").forEach(button => button.addEventListener("click", ev => {
      ev.stopPropagation();
      const box = button.nextElementSibling;
      if (box) box.hidden = !box.hidden;
    }));
    drawer.querySelector(".close").onclick = () => { view.selected = null; drawer.classList.remove("open"); apply(); };
    drawer.classList.add("open");
    apply();
  }
  function fit() {
    const xs = Object.values(boxes), rect = root.getBoundingClientRect();
    if (!xs.length) return;
    const minX = Math.min(...xs.map(b => b.l)) - 40, maxX = Math.max(...xs.map(b => b.r)) + 40;
    const minY = Math.min(...xs.map(b => b.t)) - 60, maxY = Math.max(...xs.map(b => b.b)) + 50;
    const top = 96, bottom = 46;   // clear the toolbar and the legend
    const k = Math.min(rect.width / (maxX - minX), (rect.height - top - bottom) / (maxY - minY), 1.4);
    view.k = k; view.x = -minX * k + (rect.width - (maxX - minX) * k) / 2; view.y = -minY * k + top;
    apply();
  }
  function go(step) {
    view.step = Math.max(0, Math.min(order.length - 1, step));
    select(order[view.step]);
  }
  clearInterval(view.timer);
  if (view.playing) view.timer = setInterval(tick, 2200);
  function tick() {
    if (view.step >= order.length - 1) { view.playing = false; clearInterval(view.timer); apply(); return; }
    go(view.step + 1);
  }
  bar.addEventListener("click", ev => {
    const a = ev.target.getAttribute && ev.target.getAttribute("data-a");
    if (!a) return;
    if (a === "play") {
      view.playing = !view.playing;
      clearInterval(view.timer);
      if (view.playing) { if (view.step >= order.length - 1 || view.step < 0) go(0); view.timer = setInterval(tick, 2200); }
      apply();
    } else if (a === "prev") { view.playing = false; clearInterval(view.timer); go(view.step < 0 ? 0 : view.step - 1); }
    else if (a === "next") { view.playing = false; clearInterval(view.timer); go(view.step + 1); }
    else if (a === "all") { view.playing = false; clearInterval(view.timer); view.step = -1; apply(); }
    else if (a === "fit") fit();
    else if (a === "grid" || a === "dots") {
      view.bg = a; root.className = "ato-flow " + a;
      bar.querySelector('[data-a="grid"]').classList.toggle("on", a === "grid");
      bar.querySelector('[data-a="dots"]').classList.toggle("on", a === "dots");
      apply();
    }
  });

  let drag = null;
  svg.addEventListener("mousedown", ev => { drag = {x: ev.clientX - view.x, y: ev.clientY - view.y}; svg.classList.add("dragging"); });
  window.addEventListener("mousemove", ev => { if (!drag) return; view.x = ev.clientX - drag.x; view.y = ev.clientY - drag.y; apply(); });
  window.addEventListener("mouseup", () => { drag = null; svg.classList.remove("dragging"); });
  svg.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const k = Math.max(0.3, Math.min(2.2, view.k * (ev.deltaY < 0 ? 1.1 : 0.9)));
    view.x = mx - (mx - view.x) * k / view.k; view.y = my - (my - view.y) * k / view.k; view.k = k;
    apply();
  }, {passive: false});
  svg.addEventListener("click", () => { view.selected = null; drawer.classList.remove("open"); apply(); });

  if (!view.fitted) { view.fitted = true; requestAnimationFrame(fit); }
  if (view.selected) select(view.selected); else apply();
}
"""

def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_flow_canvas", js=_JS)(**kwargs)


def render(flow: dict, key: str) -> None:
    """Draw the process flow returned by flow_model.flow()."""
    _component(data={**flow, "key": key, "css": _CSS}, key=f"flow_{key}", height=740)
