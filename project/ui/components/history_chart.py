"""Iteration histories as stacked line charts on one shared iteration axis, drawn in the browser.

Scroll zooms at the cursor, dragging across a plot zooms to that range, the bar under the plots
pans, double-click resets. The window cannot leave the recorded iterations and each y range fits
the data in view, so neither axis can be moved past the data. Hover reads every panel at one
iteration. Colours and font follow the Streamlit theme. The chart is built once per mount and
updated in place when new rows arrive, so a live run's refresh keeps a drag or hover going.
"""
import math

import streamlit as st

_JS = r"""
const states = {};   // per chart key: the iteration window and the mounted chart, kept across reruns

function ticks(lo, hi, count, integer) {
  const span = hi - lo;
  if (!(span > 0)) return {values: [lo], step: 1};
  const raw = span / Math.max(1, count), mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = Math.max(integer ? 1 : 0, mag * ((integer ? [1, 2, 5, 10] : [1, 2, 2.5, 5, 10]).find(n => n * mag >= raw * 0.75) || 10));
  const out = [];   // by index, so a tick at zero is exactly 0
  for (let i = Math.ceil(lo / step - 1e-9), last = Math.floor(hi / step + 1e-9); i <= last && out.length < 50; i++) out.push(+(i * step).toPrecision(12));
  return {values: out, step};
}

function labeler(values, step) {
  // One format per axis, with the digits the step needs: its exponent, plus one for a 2.5 x 10^k step.
  const e = Math.floor(Math.log10(step) + 1e-9);
  const extra = Number.isInteger(+(step / Math.pow(10, e)).toPrecision(6)) ? 0 : 1;
  const top = Math.max(...values.map(Math.abs));
  if (top >= 1e5 || top < 1e-3 || extra - e > 6)
    return v => v === 0 ? "0" : v.toExponential(Math.min(12, Math.max(1, Math.floor(Math.log10(Math.abs(v))) - e + extra)));
  return v => v.toFixed(Math.max(0, extra - e));
}

const readout = v => (Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3)) ? v.toExponential(4) : (+v.toPrecision(5)).toString();

export default function (component) {
  const { data, parentElement } = component;
  const key = data.key || "history";
  const view = states[key] || (states[key] = {lo: 0, hi: 1, follow: true, chart: null});
  // The runtime calls this again on the same root for every data or theme change (a live run's
  // refresh). Update in place: rebuilding would drop a drag or hover in progress.
  if (view.chart && view.chart.root !== parentElement) view.chart.dispose();
  if (!view.chart) view.chart = mount(parentElement, view);
  view.chart.update(data);
  return view.chart.dispose;   // called by the runtime on unmount
}

function mount(root, view) {
  root.innerHTML = "";
  const wrap = document.createElement("div");
  root.appendChild(wrap);
  // root is the component's ShadowRoot (isolated styles), which has no computed style: read the
  // theme from an element inside it, where the --st-* properties are inherited.
  const css = getComputedStyle(wrap);
  const theme = (name, fallback) => (css.getPropertyValue(name) || "").trim() || fallback;
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  wrap.appendChild(canvas);
  const foot = document.createElement("div");
  foot.style.cssText = "display:flex;flex-wrap:wrap;justify-content:space-between;column-gap:12px;font-size:12px;opacity:0.7;padding:2px 0 0;";
  const status = document.createElement("span"), help = document.createElement("span");
  help.textContent = "Scroll to zoom · drag across a plot to zoom to it · drag the bar to pan · double-click to reset";
  foot.append(status, help);
  wrap.appendChild(foot);

  const ctx = canvas.getContext("2d");
  const TRACK = 22, MR = 18, MT = 12, MB = 38;
  // Data, replaced by update(). xmin..xmax is the axis range: first..last, or a one-iteration
  // range centred on a single recorded iteration.
  const m = {xs: [], panels: [], H: 170, first: 0, last: 0, xmin: 0, xmax: 1, full: 1, minSpan: 1};
  let W = 0, ML = 60, hover = null, select = null, pan = null, colors = {};
  const plot = k => ({x0: ML, x1: W - MR, y0: k * m.H + MT, y1: (k + 1) * m.H - MB});
  const xpx = (v, p) => p.x0 + (v - view.lo) / (view.hi - view.lo) * (p.x1 - p.x0);
  const xval = (px, p) => view.lo + (px - p.x0) / (p.x1 - p.x0) * (view.hi - view.lo);
  const inView = v => Math.max(view.lo, Math.min(view.hi, v));
  const nearest = v => {   // index of the recorded iteration closest to v (xs is increasing)
    let a = 0, b = m.xs.length - 1;
    while (b - a > 1) { const mid = (a + b) >> 1; m.xs[mid] < v ? a = mid : b = mid; }
    return Math.abs(m.xs[a] - v) <= Math.abs(m.xs[b] - v) ? a : b;
  };
  const visible = () => {
    let a = nearest(view.lo), b = nearest(view.hi);
    if (m.xs[a] > view.lo && a > 0) a--;
    if (m.xs[b] < view.hi && b < m.xs.length - 1) b++;
    return [a, b];
  };
  const setWindow = (lo, hi) => {
    const span = Math.min(m.full, Math.max(m.minSpan, hi - lo));
    lo = Math.max(m.xmin, Math.min(lo, m.xmax - span));
    view.lo = lo; view.hi = lo + span;
    view.follow = view.lo <= m.xmin && view.hi >= m.xmax;
  };

  function update(data) {
    const xs = data.x || [];
    m.xs = xs; m.panels = data.panels || []; m.H = data.panel_height || 170;
    m.first = xs.length ? xs[0] : 0; m.last = xs.length ? xs[xs.length - 1] : m.first;
    m.xmin = xs.length > 1 ? m.first : m.first - 0.5; m.xmax = xs.length > 1 ? m.last : m.first + 0.5;
    m.full = m.xmax - m.xmin; m.minSpan = Math.min(m.full, Math.max(4, data.min_span || 10));
    if (hover !== null && hover >= xs.length) hover = null;
    if (view.follow) { view.lo = m.xmin; view.hi = m.xmax; }
    setWindow(view.lo, view.hi);
    colors = {
      text: theme("--st-text-color", css.color || "#31333f"),
      font: theme("--st-font", "'Source Sans 3', sans-serif"),
      line: (theme("--st-chart-categorical-colors", "").split(",")[0] || "").trim() || "#0068c9",
      limit: theme("--st-orange-color", "#ff8c00"),
      background: theme("--st-background-color", "#ffffff"),
      tag: theme("--st-secondary-background-color", "#f0f2f6"),
    };
    wrap.style.cssText = `position:relative;width:100%;font-family:${colors.font};color:${colors.text};user-select:none;`;
    // pan-y: a vertical swipe over the chart still scrolls the page on touch screens.
    canvas.style.cssText = `display:block;width:100%;height:${m.panels.length * m.H + TRACK}px;touch-action:pan-y;`;
    const recorded = xs.length === 1 ? `iteration ${m.first}` : `iterations ${m.first} to ${m.last}`;
    canvas.setAttribute("aria-label", `Iteration histories of ${m.panels.map(p => p.label).join(", ")}, ${recorded}`);
    draw();
    if (pan) {   // a refresh during a bar drag: continue from the refreshed window and scale
      const t = track();
      Object.assign(pan, {x: pan.cur, lo: view.lo, hi: view.hi, scale: m.full / (t.x1 - t.x0)});
    }
  }

  function yRange(values, a, b) {
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i <= b; i++) {
      const v = values[i];
      if (!Number.isFinite(v) || m.xs[i] < view.lo || m.xs[i] > view.hi) continue;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo)) return [0, 1];
    // At least a billionth of the values' size, so the range never drops below float resolution.
    const mid = Math.abs(hi + lo) / 2, pad = hi > lo ? Math.max((hi - lo) * 0.08, mid * 1e-9) : Math.max(mid * 0.01, 1e-12);
    return [lo - pad, hi + pad];
  }

  function draw() {
    W = wrap.clientWidth;
    const {text, font} = colors, n = m.panels.length, H = m.H;
    const dpr = window.devicePixelRatio || 1, total = n * H + TRACK;
    canvas.width = W * dpr; canvas.height = total * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, total);
    if (!m.xs.length || !n) return;
    ctx.font = `11px ${font}`;
    const [a, b] = visible();
    const ranges = m.panels.map(p => yRange(p.values, a, b));
    const yt = ranges.map(([lo, hi]) => {
      const t = ticks(lo, hi, Math.max(2, Math.floor((H - MT - MB) / 28)), false);
      return {...t, label: labeler(t.values, t.step)};
    });
    // One left margin for all panels so their iteration axes line up.
    ML = 30 + Math.max(24, ...yt.flatMap(t => t.values.map(v => ctx.measureText(t.label(v)).width))) + 8;
    const xt = ticks(view.lo, view.hi, Math.max(2, Math.floor((W - ML - MR) / 70)), true);

    m.panels.forEach((panel, k) => {
      const p = plot(k), [lo, hi] = ranges[k];
      const ypx = v => p.y1 - (v - lo) / (hi - lo) * (p.y1 - p.y0);
      const unit = panel.unit && panel.unit !== "–" ? ` ${panel.unit}` : "";
      ctx.fillStyle = ctx.strokeStyle = text;
      // Grid and axes
      ctx.globalAlpha = 0.12; ctx.lineWidth = 1;
      for (const v of yt[k].values) { const y = Math.round(ypx(v)) + 0.5; ctx.beginPath(); ctx.moveTo(p.x0, y); ctx.lineTo(p.x1, y); ctx.stroke(); }
      for (const v of xt.values) { const x = Math.round(xpx(v, p)) + 0.5; ctx.beginPath(); ctx.moveTo(x, p.y0); ctx.lineTo(x, p.y1); ctx.stroke(); }
      ctx.globalAlpha = 0.45;
      ctx.beginPath(); ctx.moveTo(p.x0 + 0.5, p.y0); ctx.lineTo(p.x0 + 0.5, p.y1 + 0.5); ctx.lineTo(p.x1, p.y1 + 0.5); ctx.stroke();
      ctx.globalAlpha = 0.8;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const v of yt[k].values) ctx.fillText(yt[k].label(v), p.x0 - 6, ypx(v));
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (const v of xt.values) ctx.fillText(String(v), xpx(v, p), p.y1 + 5);
      // Axis titles
      ctx.globalAlpha = 1; ctx.font = `12px ${font}`;
      ctx.fillText("Iteration", (p.x0 + p.x1) / 2, p.y1 + 20);
      ctx.save(); ctx.translate(13, (p.y0 + p.y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle";
      ctx.fillText(panel.unit ? `${panel.label} (${panel.unit})` : panel.label, 0, 0); ctx.restore();
      ctx.font = `11px ${font}`;

      ctx.save(); ctx.beginPath(); ctx.rect(p.x0, p.y0 - 1, p.x1 - p.x0, p.y1 - p.y0 + 2); ctx.clip();
      // Constraint limit: a dashed line when it is in range, otherwise a marker at the nearer edge.
      let limitTag = null;
      if (Number.isFinite(panel.limit)) {
        if (panel.limit >= lo && panel.limit <= hi) {
          const y = ypx(panel.limit);
          ctx.strokeStyle = colors.limit; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.25;
          ctx.beginPath(); ctx.moveTo(p.x0, y); ctx.lineTo(p.x1, y); ctx.stroke(); ctx.setLineDash([]);
          limitTag = {text: `limit ${readout(panel.limit)}${unit}`, y: y - p.y0 > 20 ? y - 10 : y + 10};
        } else {
          const above = panel.limit > hi;
          limitTag = {text: `limit ${readout(panel.limit)}${unit} ${above ? "↑" : "↓"}`, y: above ? p.y0 + 10 : p.y1 - 10};
        }
      }
      // History line, with markers once few enough points are in view to tell them apart.
      ctx.strokeStyle = ctx.fillStyle = colors.line; ctx.lineWidth = 1.75; ctx.lineJoin = "round";
      ctx.beginPath();
      let pen = false;
      for (let i = a; i <= b; i++) {
        const v = panel.values[i];
        if (!Number.isFinite(v)) { pen = false; continue; }
        const x = xpx(m.xs[i], p), y = ypx(v);
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
      }
      ctx.stroke();
      if ((p.x1 - p.x0) / Math.max(1, b - a) > 9) {
        for (let i = a; i <= b; i++) if (Number.isFinite(panel.values[i])) {
          ctx.beginPath(); ctx.arc(xpx(m.xs[i], p), ypx(panel.values[i]), 2.25, 0, 2 * Math.PI); ctx.fill();
        }
      }
      if (limitTag) {   // drawn over the line, on a backing, so it stays readable
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        const tw = ctx.measureText(limitTag.text).width + 8;
        ctx.fillStyle = colors.background; ctx.globalAlpha = 0.85; ctx.fillRect(p.x1 - 2 - tw, limitTag.y - 8, tw, 16);
        ctx.globalAlpha = 1; ctx.fillStyle = colors.limit; ctx.fillText(limitTag.text, p.x1 - 6, limitTag.y);
      }
      // Range being selected
      if (select) {
        const s0 = Math.min(select.from, select.to), s1 = Math.max(select.from, select.to);
        ctx.globalAlpha = 0.14; ctx.fillStyle = colors.line;
        ctx.fillRect(xpx(s0, p), p.y0, xpx(s1, p) - xpx(s0, p), p.y1 - p.y0);
        ctx.globalAlpha = 1;
      }
      // Hover read-out
      if (hover !== null && !select) {
        const x = xpx(m.xs[hover], p), v = panel.values[hover];
        ctx.strokeStyle = text; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, p.y0); ctx.lineTo(Math.round(x) + 0.5, p.y1); ctx.stroke();
        ctx.globalAlpha = 1;
        if (Number.isFinite(v)) {
          ctx.fillStyle = colors.line; ctx.beginPath(); ctx.arc(x, ypx(v), 3.5, 0, 2 * Math.PI); ctx.fill();
          const tag = `iteration ${m.xs[hover]}: ${readout(v)}${unit}`;
          ctx.font = `12px ${font}`;
          const tw = ctx.measureText(tag).width + 12, left = x + 8 + tw > p.x1 ? x - 8 - tw : x + 8;
          ctx.fillStyle = colors.tag; ctx.globalAlpha = 0.92;
          ctx.fillRect(left, p.y0 + 2, tw, 20); ctx.globalAlpha = 1;
          ctx.fillStyle = text; ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(tag, left + 6, p.y0 + 12);
          ctx.font = `11px ${font}`;
        }
      }
      ctx.restore();
    });

    // Window bar: the whole run as a track, the iterations in view as a thumb that pans.
    const t = track();
    ctx.fillStyle = text; ctx.globalAlpha = 0.1;
    roundRect(t.x0, t.y, t.x1 - t.x0, t.h, t.h / 2); ctx.fill();
    ctx.globalAlpha = view.follow ? 0.25 : 0.55;
    roundRect(t.t0, t.y, Math.max(8, t.t1 - t.t0), t.h, t.h / 2); ctx.fill();
    ctx.globalAlpha = 1;
    status.textContent = m.xs.length === 1 ? `1 recorded iteration (${m.first})`
      : view.follow ? `All ${m.xs.length} recorded iterations (${m.first}–${m.last})`
      : `Iterations ${Math.round(view.lo)}–${Math.round(view.hi)} of ${m.first}–${m.last}`;
  }

  function track() {
    const p = plot(0), y = m.panels.length * m.H + 6, h = 8;
    const t0 = p.x0 + (view.lo - m.xmin) / m.full * (p.x1 - p.x0), t1 = p.x0 + (view.hi - m.xmin) / m.full * (p.x1 - p.x0);
    return {x0: p.x0, x1: p.x1, y, h, t0, t1};
  }
  function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }

  const at = ev => {
    const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top, n = m.panels.length;
    const k = Math.floor(y / m.H), p = plot(Math.min(Math.max(k, 0), n - 1));
    return {x, y, p, inPlot: k >= 0 && k < n && x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1,
            inTrack: y >= n * m.H && x >= p.x0 - 4 && x <= p.x1 + 4};
  };

  canvas.addEventListener("wheel", ev => {
    const h = at(ev);
    if (!h.inPlot || !m.xs.length) return;
    const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1;   // lines or pages to pixels
    const dx = ev.deltaX * unit, dy = ev.deltaY * unit, lo = view.lo, hi = view.hi;
    if (ev.shiftKey || Math.abs(dx) > Math.abs(dy)) {   // pan; some systems deliver Shift+wheel as dx
      const d = (Math.abs(dx) > Math.abs(dy) ? dx : dy) / (h.p.x1 - h.p.x0) * (hi - lo);
      setWindow(lo + d, hi + d);
    } else {   // zoom at the cursor
      const anchor = xval(h.x, h.p), span = hi - lo;
      const next = Math.min(m.full, Math.max(m.minSpan, span * Math.exp(Math.max(-1, Math.min(1, dy * 0.002)))));
      setWindow(anchor - (anchor - lo) * next / span, anchor + (hi - anchor) * next / span);
    }
    const tol = 1e-9 * Math.max(1, m.full);
    if (view.follow && Math.abs(view.lo - lo) <= tol && Math.abs(view.hi - hi) <= tol) {
      // Showing every iteration and the wheel changes nothing: let the page scroll, but never pinch-zoom it.
      if (ev.ctrlKey) ev.preventDefault();
      return;
    }
    ev.preventDefault();
    draw();
  }, {passive: false});

  canvas.addEventListener("pointerdown", ev => {
    if (ev.button !== 0 || !m.xs.length) return;
    const h = at(ev);
    if (h.inTrack) {
      if (view.follow) return;   // every iteration is in view: nothing to pan
      const t = track(), start = {lo: view.lo, hi: view.hi};
      if (h.x < t.t0 || h.x > t.t1) {   // click on the track: centre the window there
        const c = m.xmin + (h.x - t.x0) / (t.x1 - t.x0) * m.full, span = view.hi - view.lo;
        setWindow(c - span / 2, c + span / 2);
      }
      pan = {x: h.x, cur: h.x, lo: view.lo, hi: view.hi, scale: m.full / (t.x1 - t.x0), start};
    } else if (h.inPlot) {
      const v = inView(xval(h.x, h.p));
      select = {from: v, to: v, x: h.x};
    } else return;
    canvas.setPointerCapture(ev.pointerId);
    draw();
  });
  canvas.addEventListener("pointermove", ev => {
    const h = at(ev);
    if (pan) {
      pan.cur = h.x;
      const d = (h.x - pan.x) * pan.scale;
      setWindow(pan.lo + d, pan.hi + d);
    } else if (select) {
      select.to = inView(xval(h.x, h.p));   // the shaded range is what gets zoomed to
    } else {
      hover = h.inPlot && m.xs.length ? nearest(xval(h.x, h.p)) : null;
      canvas.style.cursor = h.inTrack ? (view.follow ? "default" : "grab") : h.inPlot ? "crosshair" : "default";
    }
    draw();
  });
  canvas.addEventListener("pointerup", ev => {
    if (select) {
      if (Math.abs(at(ev).x - select.x) > 4) setWindow(Math.min(select.from, select.to), Math.max(select.from, select.to));
      select = null;
    }
    pan = null;
    draw();
  });
  // A cancelled gesture (scroll taken over by the browser, palm rejection) changes nothing.
  canvas.addEventListener("pointercancel", () => {
    if (pan) setWindow(pan.start.lo, pan.start.hi);
    select = null; pan = null; draw();
  });
  canvas.addEventListener("pointerleave", () => { if (!select && !pan) { hover = null; draw(); } });
  canvas.addEventListener("dblclick", () => { setWindow(m.xmin, m.xmax); draw(); });

  const resize = new ResizeObserver(() => draw());
  resize.observe(wrap);
  const chart = {root, update, dispose: () => { resize.disconnect(); if (view.chart === chart) view.chart = null; }};
  return chart;
}
"""


def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_history_chart", js=_JS)(**kwargs)


def render(iterations: list[int], panels: list[dict], *, key: str, panel_height: int = 170) -> None:
    """iterations: recorded iteration numbers, increasing. panels: one dict per plot with "label",
    "values" (aligned with iterations), optional "unit" (axis title suffix) and "limit" (dashed line).
    key: one per run, so a zoom belongs to the run it was made on."""
    # NaN or inf (a diverging run) is not valid JSON; send a gap instead.
    panels = [{**panel, "values": [v if math.isfinite(v) else None for v in panel["values"]]} for panel in panels]
    # Height follows the content (default): the canvas has a fixed CSS height and the footer may wrap.
    _component(data={"x": iterations, "panels": panels, "key": key, "panel_height": panel_height},
               key=f"history_{key}")
