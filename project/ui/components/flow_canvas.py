"""Process-flow canvas: flowchart symbols on a pannable grid, a detail drawer per node, and playback.

Everything interactive runs in the browser; Python only passes the flow from flow_model.flow().
While a formulation step or its run is in progress the page passes a fresh flow every few
seconds: the canvas updates in place, marks the active node, and keeps pan, zoom and the
open drawer.
"""
import functools
import json
import logging

import streamlit as st

from project.ui import api, flow_model

_log = logging.getLogger(__name__)

_CSS = """
.ato-flow { position: relative; border: 1px solid #d6d6d6; border-radius: 10px; overflow: hidden;
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
.ato-step { font-size: 13px; color: #555; background: #ffffffcc; padding: 3px 8px; border-radius: 6px; }
.ato-live, .ato-status { font-size: 12.5px; padding: 3px 9px; border-radius: 999px; border: 1px solid; }
.ato-live { background: #e8f1fb; border-color: #9cc3e8; color: #0b4f8a; }
.ato-live::before { content: "●"; color: #0b6bcb; margin-right: 5px; animation: ato-blink 1.2s ease-in-out infinite; }
.ato-status { background: #fff; border-color: #d6d6d6; color: #444; }
@keyframes ato-blink { 50% { opacity: .25; } }
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
.node.active .shape { stroke: #0b6bcb; stroke-width: 2.8; fill: #eef5fc; animation: ato-pulse 1.4s ease-in-out infinite; }
.node.waiting .shape { stroke: #b7791f; stroke-width: 2; stroke-dasharray: 6 4; fill: #fffaf0; }
@keyframes ato-pulse { 0%, 100% { filter: drop-shadow(0 0 1px #0b6bcb66); } 50% { filter: drop-shadow(0 0 9px #0b6bcb); } }
.node.hidden, .edge.hidden { opacity: 0.08; }
.edge path { fill: none; stroke: #2b2b2b; stroke-width: 2.4; }
.edge.dashed path { stroke-dasharray: 7 5; stroke: #777; }
.edge.flowing path { stroke: #0b6bcb; stroke-dasharray: 9 6; animation: ato-march .8s linear infinite; }
@keyframes ato-march { to { stroke-dashoffset: -15; } }
.edge text { font-size: 12px; font-weight: 600; fill: #2b2b2b; }
.edge { transition: opacity .25s; }
.ato-legend { position: absolute; bottom: 10px; left: 10px; right: 10px; display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11.5px; background: #ffffffd9;
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
const views = {};    // per canvas key: pan, zoom, selection, playback, follow
const mounts = {};   // per canvas key: the drawn canvas, so a live refresh can update it in place

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
function aboutFor(label) {
  // Step descriptions are keyed by node label; "Specification v2" uses "Specification", and so on.
  const exact = STATIC.about[label];
  if (exact !== undefined) return exact;
  const hit = Object.entries(STATIC.about).find(([k]) => String(label).startsWith(k));
  return hit ? hit[1] : null;
}
function promptsHtml(d) {
  if (!d.prompts || !d.prompts.length) return "";
  return `<details open><summary>Prompts sent (${d.prompts.length} parts)</summary>` +
    (d.prompt_note ? `<p class="rule">${esc(d.prompt_note)}</p>` : "") +
    d.prompts.map(p => `<details><summary>${esc(p.name)}</summary>${prettyPrompt(STATIC.prompts[p.name])}</details>`).join("") + "</details>";
}
function block(title, text, open) {
  return text ? `<details ${open ? "open" : ""}><summary>${esc(title)}</summary><pre>${esc(text)}</pre></details>` : "";
}
function detailHtml(n) {
  const d = n.detail || {};
  let html = `<button class="close" title="Close">×</button><h3>${esc(n.label)}</h3>
    <div class="meta">${esc(n.shape)} · ${esc(n.actor)}${n.sub ? " · " + esc(n.sub) : ""}${n.status !== "ok" ? " · " + esc(n.status) : ""}</div>`;
  const about = aboutFor(n.label);
  if (about) html += `<div class="about"><p>${esc(about.text)}${citeHtml(about.cite)}</p><ul>` +
    (about.points || []).map(x => `<li>${esc(x.text)}${citeHtml(x.cite)}</li>`).join("") + `</ul></div>`;
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

// The structure of a flow: what needs a rebuild when it changes. Status, subtitles, details
// and the live line change in place.
function structure(flow) {
  return JSON.stringify([
    flow.height || 720,
    flow.nodes.map(n => [n.id, n.col, n.row, n.shape, n.actor, n.label, !!n.sub,
                         ((n.detail || {}).prompts || []).map(p => p.name)]),
    flow.edges.map(e => [e.from, e.to, e.label || "", e.exit || ""]),
  ]);
}

export default function (component) {
  const { data, parentElement } = component;
  const flow = data || {nodes: [], edges: []};
  const key = flow.key || "flow";
  const view = views[key] || (views[key] = {x: 20, y: 30, k: 0.78, bg: "grid", selected: null, step: -1,
                                            playing: false, follow: true, lastActive: null});
  const sig = structure(flow);
  const old = mounts[key];
  if (old && old.parent === parentElement && old.root.isConnected && old.sig === sig) {
    old.update(flow, false);
    return old.unmount;   // Streamlit keeps the latest returned function and calls it on unmount
  }
  const saved = old && old.root.isConnected ? old.drawerState() : null;
  if (old) old.teardown();
  let current = flow, drawerSig = null;

  parentElement.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = STATIC.css;
  parentElement.appendChild(style);
  const root = document.createElement("div");
  root.className = "ato-flow " + view.bg;
  root.style.height = (flow.height || 720) + "px";
  parentElement.appendChild(root);

  const bar = document.createElement("div");
  bar.className = "ato-bar";
  bar.innerHTML = `<span class="ato-live" style="display:none"></span>
    <button data-a="play">▶ Play</button><button data-a="prev">◀</button><button data-a="next">▶|</button>
    <button data-a="all">Show all</button><button data-a="fit">Fit</button>
    <button data-a="follow" title="Keep the active step in view">Follow</button>
    <button data-a="grid" class="${view.bg === "grid" ? "on" : ""}">Grid</button><button data-a="dots" class="${view.bg === "dots" ? "on" : ""}">Dots</button>
    <span class="ato-step"></span>`;
  root.appendChild(bar);
  const liveEl = bar.querySelector(".ato-live"), followBtn = bar.querySelector('[data-a="follow"]');
  const legend = document.createElement("div");
  legend.className = "ato-legend";
  legend.innerHTML = '<span><i style="border-color:#6d4fc2"></i>model</span><span><i style="border-color:#2b2b2b"></i>code</span>' +
    '<span><i style="border-color:#b7791f"></i>engineer</span><span><i style="border-color:#c62828;background:#fdecea"></i>failed</span>' +
    '<span><i style="border-color:#0b6bcb;background:#eef5fc"></i>running now</span><span><i style="border-color:#777;border-style:dashed"></i>still to come</span>' +
    '<span>▭ process ▱ data ⬭ review document ◇ decision 🗀 record</span>';
  root.appendChild(legend);
  const drawer = document.createElement("div");
  drawer.className = "ato-drawer" + (view.selected ? " open" : "");   // already open: no slide-in on a rebuild
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
      (node.detail.prompt_note ? `<p class="rule">${esc(node.detail.prompt_note)}</p>` : "") + prettyPrompt(STATIC.prompts[part.name]);
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

  const boxes = {}, nodeEls = {}, subEls = {}, edgeEls = [];
  const nodeClass = n => `node ${n.shape} ${n.actor} ${n.status}`;
  const edgeClass = (e, f) => {
    const target = f.nodes.find(n => n.id === e.to);
    return "edge" + (e.dashed ? " dashed" : "") + (target && target.status === "active" ? " flowing" : "");
  };
  flow.nodes.forEach(n => { boxes[n.id] = box(n); });
  flow.edges.forEach((e, index) => {
    const a = boxes[e.from], b = boxes[e.to];
    if (!a || !b) return;
    const pts = route(a, b, e.exit);
    const ge = el("g", {class: edgeClass(e, flow)}, g);
    el("path", {d: "M" + pts.map(p => p.join(",")).join(" L"), "marker-end": `url(#arrow-${key})`}, ge);
    if (e.label) {
      const [p0, p1] = pts;
      const horizontal = Math.abs(p0[1] - p1[1]) < 1;
      const leftward = horizontal && p1[0] < p0[0];
      const t = el("text", {x: p0[0] + (leftward ? -34 : horizontal ? 10 : 8), y: p0[1] + (horizontal ? -8 : 16)}, ge);
      t.textContent = e.label;
    }
    edgeEls.push({ge, e, index});
  });
  flow.nodes.forEach(n => {
    const b = boxes[n.id];
    const gn = el("g", {class: nodeClass(n)}, g);
    el("path", {class: "shape", d: shapePath(n, b)}, gn);
    const label = el("text", {x: b.x, y: b.y - (n.sub ? 7 : 0)}, gn);
    label.textContent = n.label;
    if (n.sub) { subEls[n.id] = el("text", {class: "sub", x: b.x, y: b.y + 11}, gn); subEls[n.id].textContent = n.sub; }
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
        pg.addEventListener("mouseenter", ev => showPrompt(part, current.nodes.find(x => x.id === n.id) || n, ev));
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
    if (revealed) current.nodes.forEach(n => { if (n.status === "active" || n.status === "waiting") revealed.add(n.id); });
    for (const id in nodeEls) {
      nodeEls[id].classList.toggle("hidden", !!revealed && !revealed.has(id));
      nodeEls[id].classList.toggle("selected", view.selected === id);
      nodeEls[id].classList.toggle("current", view.step >= 0 && order[view.step] === id);
    }
    edgeEls.forEach(({ge, e}) => ge.classList.toggle("hidden", !!revealed && !(revealed.has(e.from) && revealed.has(e.to))));
    stepLabel.textContent = view.step < 0 ? `${order.length} steps` : `Step ${view.step + 1} of ${order.length}`;
    bar.querySelector('[data-a="play"]').textContent = view.playing ? "❚❚ Pause" : "▶ Play";
    followBtn.classList.toggle("on", view.follow);
  }
  function drawerState() {
    // The reader's place in the drawer: scroll, which sections are open, which code boxes are shown.
    return {scroll: drawer.scrollTop,
            open: Object.fromEntries([...drawer.querySelectorAll("details")].map(d => [d.querySelector("summary")?.textContent, d.open])),
            code: [...drawer.querySelectorAll(".codebox")].map(b => b.hidden),
            pre: [...drawer.querySelectorAll("pre")].map(p => p.scrollTop)};
  }
  function renderDrawer(n, keep) {
    // keep: true keeps the place in this drawer; an object restores a place saved before a rebuild.
    const state = keep === true ? drawerState() : keep || null;
    drawer.innerHTML = detailHtml(n);
    drawer.querySelectorAll(".cite").forEach(button => button.addEventListener("click", ev => {
      ev.stopPropagation();
      const box = button.nextElementSibling;
      if (box) box.hidden = !box.hidden;
    }));
    drawer.querySelector(".close").onclick = () => { view.selected = null; drawer.classList.remove("open"); apply(); };
    if (state) {
      drawer.querySelectorAll("details").forEach(d => {
        const s = d.querySelector("summary")?.textContent;
        if (s in state.open) d.open = state.open[s];
      });
      drawer.querySelectorAll(".codebox").forEach((b, i) => { if (i < state.code.length) b.hidden = state.code[i]; });
      drawer.querySelectorAll("pre").forEach((p, i) => { if (i < state.pre.length) p.scrollTop = state.pre[i]; });
      drawer.scrollTop = state.scroll;
    }
    drawerSig = JSON.stringify([n.detail, n.status, n.sub]);
  }
  function select(id) {
    view.selected = id;
    const n = current.nodes.find(x => x.id === id);
    if (!n) { drawer.classList.remove("open"); apply(); return; }
    renderDrawer(n, false);
    drawer.classList.add("open");
    apply();
  }
  // Where a focused step sits vertically: below the toolbar, in the upper part of the canvas.
  const focusY = rect => Math.min((rect.height + 50) / 2, 280);
  function reveal(id) {
    // Pan just enough to bring the active step back into view; leave the zoom alone.
    const b = boxes[id], rect = root.getBoundingClientRect();
    if (!b || !rect.width) return;
    const right = rect.width - (drawer.classList.contains("open") ? drawer.offsetWidth : 0);
    const sx = view.x + b.x * view.k, sy = view.y + b.y * view.k;
    if (sx < 90 || sx > right - 90 || sy < 120 || sy > rect.height - 70) {
      view.x = right / 2 - b.x * view.k;
      view.y = focusY(rect) - b.y * view.k;
      apply();
    }
  }
  function fit(minK = 0) {
    // The whole flow; with minK, never smaller than that, centred on the step that matters now.
    const xs = Object.values(boxes), rect = root.getBoundingClientRect();
    if (!xs.length) return;
    const minX = Math.min(...xs.map(b => b.l)) - 40, maxX = Math.max(...xs.map(b => b.r)) + 40;
    const minY = Math.min(...xs.map(b => b.t)) - 60, maxY = Math.max(...xs.map(b => b.b)) + 50;
    const top = 96, bottom = 46;   // clear the toolbar and the legend
    const k = Math.min(rect.width / (maxX - minX), (rect.height - top - bottom) / (maxY - minY), 1.4);
    view.k = Math.max(k, minK); view.x = -minX * view.k + (rect.width - (maxX - minX) * view.k) / 2; view.y = -minY * view.k + top;
    if (view.k > k) {
      const focus = boxes[current.active] || boxes[order.filter(id => !/^(next|live):/.test(id)).slice(-1)[0]];
      if (focus) { view.x = rect.width / 2 - focus.x * view.k; view.y = focusY(rect) - focus.y * view.k; }
    }
    apply();
  }
  function update(next, first) {
    current = next;
    for (const n of next.nodes) {
      if (!nodeEls[n.id]) continue;
      nodeEls[n.id].setAttribute("class", nodeClass(n));
      if (subEls[n.id] && subEls[n.id].textContent !== n.sub) subEls[n.id].textContent = n.sub || "";
    }
    edgeEls.forEach(({ge, index}) => ge.setAttribute("class", edgeClass(next.edges[index], next)));
    liveEl.textContent = next.status_text || (next.live ? "Live" : "");
    liveEl.className = next.live ? "ato-live" : "ato-status";
    liveEl.style.display = liveEl.textContent ? "" : "none";
    followBtn.style.display = next.live ? "" : "none";
    if (view.selected) {
      const n = next.nodes.find(x => x.id === view.selected);
      if (!n) { view.selected = null; drawer.classList.remove("open"); }
      else if (first) { renderDrawer(n, saved); drawer.classList.add("open"); }
      else if (JSON.stringify([n.detail, n.status, n.sub]) !== drawerSig) renderDrawer(n, true);
    }
    if (next.active && view.follow && view.fitted && (first || next.active !== view.lastActive)) reveal(next.active);
    view.lastActive = next.active;
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
    else if (a === "fit") fit();   // everything, however small
    else if (a === "follow") { view.follow = !view.follow; if (view.follow && current.active) reveal(current.active); apply(); }
    else if (a === "grid" || a === "dots") {
      view.bg = a; root.className = "ato-flow " + a;
      bar.querySelector('[data-a="grid"]').classList.toggle("on", a === "grid");
      bar.querySelector('[data-a="dots"]').classList.toggle("on", a === "dots");
      apply();
    }
  });

  // Dragging pans the canvas and stops following the active step (Follow turns it back on).
  let drag = null;
  svg.addEventListener("mousedown", ev => {
    drag = {x: ev.clientX - view.x, y: ev.clientY - view.y, sx: ev.clientX, sy: ev.clientY, moved: false};
    svg.classList.add("dragging");
  });
  const onMove = ev => {
    // A click with a little hand jitter is a click, not a drag that stops Follow.
    if (!drag) return;
    if (!drag.moved) {
      if (Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) <= 4) return;
      drag.moved = true; view.follow = false;
    }
    view.x = ev.clientX - drag.x; view.y = ev.clientY - drag.y;
    apply();
  };
  const onUp = () => { drag = null; svg.classList.remove("dragging"); };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  svg.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const k = Math.max(0.3, Math.min(2.2, view.k * (ev.deltaY < 0 ? 1.1 : 0.9)));
    view.x = mx - (mx - view.x) * k / view.k; view.y = my - (my - view.y) * k / view.k; view.k = k;
    apply();
  }, {passive: false});
  svg.addEventListener("click", () => { view.selected = null; drawer.classList.remove("open"); apply(); });

  // Fit once the canvas has a size: it has none while it sits in a collapsed expander.
  const sizer = new ResizeObserver(() => {
    if (view.fitted || !root.clientWidth) return;
    fit(0.55);
    view.fitted = true;
    if (current.active && view.follow) reveal(current.active);
  });
  sizer.observe(root);
  // A rebuild drops this drawing's listeners; unmounting (Streamlit calls the returned function) also stops playback.
  const teardown = () => { sizer.disconnect(); window.removeEventListener("mousemove", onMove);
                           window.removeEventListener("mouseup", onUp); };
  const mount = {parent: parentElement, root, sig, update, drawerState, teardown};
  mount.unmount = () => {
    teardown();
    clearInterval(view.timer);
    view.playing = false;
    if (mounts[key] === mount) delete mounts[key];
  };
  mounts[key] = mount;
  update(flow, true);
  return mount.unmount;
}
"""

@functools.lru_cache(maxsize=1)
def _script() -> str:
    # What never changes while the app runs travels in the script, not with every live refresh:
    # the stylesheet, the step descriptions with their code citations, and the prompt texts.
    static = {"css": _CSS, "about": flow_model.about_table(), "prompts": flow_model.prompt_texts()}
    return "const STATIC = " + json.dumps(static) + ";\n" + _JS


def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_flow_canvas", js=_script())(**kwargs)


def render(flow: dict, key: str, height: int = 720) -> None:
    """Draw the process flow returned by flow_model.flow()."""
    _component(data={**flow, "key": key, "height": height}, key=f"flow_{key}", height=height + 20)


def show(fid: str, key: str, height: int = 720, flow: dict | None = None) -> None:
    """A formulation's process flow; while a step or its run is in progress, refreshed in place every 2 s."""
    if api.is_live(fid):
        _live(fid, key, height)
    else:
        _draw(fid, key, height, flow)


@st.fragment(run_every=2)
def _live(fid: str, key: str, height: int) -> None:
    flow = _draw(fid, key, height)
    if flow is not None and not flow["live"]:
        st.rerun()   # the whole page: a finished flow is drawn once, without refreshing


def _draw(fid: str, key: str, height: int, flow: dict | None = None) -> dict | None:
    # A flow that cannot be built (say, a damaged run folder) must not take the page down with it.
    try:
        flow = flow or flow_model.flow(fid)
    except Exception:
        _log.exception("Process flow for %s could not be built", fid)
        st.error("The process flow could not be drawn for this formulation.")
        return None
    render(flow, key, height)
    return flow
