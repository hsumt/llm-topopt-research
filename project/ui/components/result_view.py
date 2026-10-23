"""Orbitable 3D view of a 2D level-set design: one cube per element with material, one element deep.

The solver is plane stress; the extrusion is only for viewing. The view follows the ParaView renders
(render_paraview.py): y up, orthographic, one scale for every view, element colours from the inverted
ColorBrewer Blues ramp, and the same standard views, picked from the view_hud overlay or keys 1-7.
Drag to orbit (azimuth about y, elevation about the screen's x axis), scroll to zoom, double-click for
the isometric view. Given the solver config, the clamp and the load are drawn where the solver applies
them. The chrome follows the Streamlit theme. The viewer is built once per mount and updated in place,
so a rerun or a theme change keeps a drag going; the camera is kept per key.
"""
import json

import streamlit as st

from project.ui.components import view_hud

# LBracket2DConfig fields sent with the overlays (the clamp along the top of the vertical arm, the load on the
# tip band); the script reads lx_m, load_patch_m and force_n, the rest record what they were drawn from.
_OVERLAY_FIELDS = ("lx_m", "ly_m", "cut_length_m", "load_patch_m", "force_n", "nelx")

_CSS = r"""
.rv { display: flex; flex-direction: column; height: 520px; border: 1px solid var(--st-border-color, #d6d6d6);
      border-radius: 10px; overflow: hidden; user-select: none; color: var(--st-text-color, #1f2328);
      font-family: var(--st-font, 'Source Sans 3', system-ui, sans-serif); }
/* The stage mirrors paraview_view: view badge top left, view panel top right, the model centred. */
.stage { position: relative; flex: 1 1 auto; min-height: 0; outline: none;
         background: radial-gradient(circle at 50% 45%, var(--st-background-color, #ffffff), var(--st-secondary-background-color, #f3f5f8)); }
/* Dark: lighter behind the model, so the dark end of the ramp stays apart from the page. */
.rv.dark .stage { background: radial-gradient(circle at 50% 45%, var(--st-secondary-background-color, #1b1f27), var(--st-background-color, #0e1117)); }
.stage:focus-visible { box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--st-primary-color, #1f4e8c) 60%, transparent); }
.stage canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
.stage canvas.dragging { cursor: grabbing; }
.probe { display: none; }
.chip { position: absolute; left: 10px; box-sizing: border-box; padding: 3px 8px; border-radius: 6px;
        background: color-mix(in srgb, var(--st-background-color, #ffffff) 85%, transparent);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
.chip b { font-weight: 600; }
.rv-at { top: 10px; font-size: 13px; max-width: calc(100% - 100px); }
.stage.open .rv-at { max-width: calc(100% - 246px); }   /* stops short of the open view panel */
.legend { left: auto; right: 10px; bottom: 8px; font-size: 12px; max-width: calc(100% - 20px); }   /* where ParaView puts its bar */
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; padding: 6px 10px; font-size: 13px;
       border-top: 1px solid var(--st-border-color, #d6d6d6); background: var(--st-secondary-background-color, #f3f5f8); }
.rv-title { flex: 0 1 auto; margin-right: auto; min-width: 0; font-weight: 600; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }
.bar label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; white-space: nowrap; }
.bar label[hidden] { display: none; }
.bar input { margin: 0; accent-color: var(--st-primary-color, #1f4e8c); cursor: pointer; }
.hint { flex: 1 1 100%; font-size: 12px; color: color-mix(in srgb, var(--st-text-color, #1f2328) 70%, transparent); }
.ramp { display: inline-block; width: 64px; height: 10px; vertical-align: -1px; margin: 0 5px; border-radius: 2px;
        border: 1px solid var(--st-border-color, #9aa7b5);
        background: linear-gradient(90deg, #f7fbff, #deebf7, #c6dbef, #9ecae1, #6baed6, #4292c6, #2171b5, #08519c, #08306b); }
"""

_MODULE = r"""
const states = {};   // per viewer key: camera, toggles and the mounted viewer, kept across reruns
const RAD = Math.PI / 180;
// ParaView's "Blues" preset inverted, as render_paraview.py colours elements: density 0 -> 1, nine stops.
const RAMP = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
  .map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
// [azimuth about y, elevation] in degrees; azimuth 0, elevation 0 puts the camera on +z (the front view).
// The same directions and screen-up axes as render_paraview.py VIEWS.
const STANDARD = {front: [0, 0], back: [180, 0], left: [-90, 0], right: [90, 0], top: [0, 90], bottom: [0, -90],
                  isometric: [45, Math.atan(Math.SQRT1_2) / RAD]};
const LIGHT = (v => v.map(c => c / Math.hypot(...v)))([0.3, 0.45, 0.84]);   // view space: upper right, toward the viewer

const ramp = d => { const t = Math.max(0, Math.min(1, d)) * 8, i = Math.min(7, Math.floor(t)), f = t - i;
  return RAMP[i].map((c, n) => c + (RAMP[i + 1][n] - c) * f); };
const wrapAz = a => { a = ((a + 180) % 360 + 360) % 360 - 180; return a === -180 ? 180 : a; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const fmt = v => String(+(+v).toPrecision(4)).replace("-", "−");
// y-up turntable camera: screen right r, screen up u, and d, the direction from the model toward the camera.
function camera(az, el) {
  const a = az * RAD, e = el * RAD, ca = Math.cos(a), sa = Math.sin(a), ce = Math.cos(e), se = Math.sin(e);
  return {r: [ca, 0, -sa], u: [-se * sa, ce, -se * ca], d: [ce * sa, se, ce * ca]};
}

export default function (component) {
  const { data, parentElement } = component;
  const key = data.key || "result";
  const view = states[key] || (states[key] = {az: STANDARD.isometric[0], el: STANDARD.isometric[1], zoom: 1,
                                               ghost: true, marks: true, viewer: null});
  // The runtime calls this again on the same root for every data or theme change: update in place.
  if (view.viewer && view.viewer.root !== parentElement) view.viewer.dispose();
  if (!view.viewer) view.viewer = mount(parentElement, view);
  view.viewer.update(data);
  return view.viewer.dispose;   // called by the runtime on unmount
}

function mount(root, view) {
  root.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = CSS;   // in the module, not registered: clearing the root would drop a registered sheet
  const wrap = document.createElement("div");
  wrap.className = "rv";
  const stage = document.createElement("div");
  stage.className = "stage"; stage.tabIndex = 0;
  stage.setAttribute("aria-label", "Design viewer. Drag to orbit, scroll to zoom, double-click for the isometric view. "
    + "Keys 1 to 7 select front, back, left, right, top, bottom and isometric views.");
  stage.innerHTML = `<canvas role="img"></canvas><i class="probe"></i><div class="chip rv-at"></div>
    <div class="chip legend">element density<span class="ramp"></span>0 → 1</div>`;
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<span class="rv-title"></span>
    <label><input type="checkbox" data-toggle="ghost">design-space outline</label>
    <label><input type="checkbox" data-toggle="marks">supports &amp; load</label>
    <span class="hint">Drag to orbit · scroll to zoom · double-click for isometric · keys 1–7 for the standard views ·
      one element deep for viewing (the solver is plane stress)</span>`;
  wrap.append(stage, bar);
  root.append(style, wrap);
  const canvas = stage.querySelector("canvas"), probe = stage.querySelector(".probe"), at = stage.querySelector(".rv-at");
  const title = bar.querySelector(".rv-title"), marksToggle = bar.querySelector("[data-toggle=marks]").parentElement;
  for (const input of bar.querySelectorAll("input")) {
    input.checked = view[input.dataset.toggle];
    input.addEventListener("change", () => { view[input.dataset.toggle] = input.checked; draw(); });
  }

  const m = {nx: 0, ny: 0, radius: 1, quads: [], outline: [], marks: null};   // geometry, centred on the domain
  let theme = {}, shown, anim = 0, timer = 0, drag = null;
  const current = () => Object.keys(STANDARD).find(name => Math.abs(wrapAz(view.az) - wrapAz(STANDARD[name][0])) < 1e-6
    && Math.abs(view.el - STANDARD[name][1]) < 1e-6) || null;
  const hud = viewHud(stage, {state: view, available: () => true, current, select: snap, missing: () => ""});

  function build(data) {
    const grid = data.density || [], ny = grid.length, nx = ny ? grid[0].length : 0, cut = data.cut_cells || 0;
    const cx = nx / 2, cy = ny / 2, quads = [];
    // Every element with density above 0.001 is a cube (no 0.5 cut-off, no smoothing), as ParaView's threshold.
    const solid = (i, j) => i >= 0 && j >= 0 && i < ny && j < nx && grid[i][j] > 0.001;
    const add = (p, n, c) => quads.push({p, n, c, m: [0, 1, 2].map(k => (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4)});
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      if (!solid(i, j)) continue;
      const c = ramp(grid[i][j]), x0 = j - cx, x1 = x0 + 1, y0 = i - cy, y1 = y0 + 1;
      add([[x0, y0, .5], [x1, y0, .5], [x1, y1, .5], [x0, y1, .5]], [0, 0, 1], c);
      add([[x0, y0, -.5], [x0, y1, -.5], [x1, y1, -.5], [x1, y0, -.5]], [0, 0, -1], c);
      if (!solid(i, j - 1)) add([[x0, y0, -.5], [x0, y0, .5], [x0, y1, .5], [x0, y1, -.5]], [-1, 0, 0], c);
      if (!solid(i, j + 1)) add([[x1, y0, -.5], [x1, y1, -.5], [x1, y1, .5], [x1, y0, .5]], [1, 0, 0], c);
      if (!solid(i - 1, j)) add([[x0, y0, -.5], [x1, y0, -.5], [x1, y0, .5], [x0, y0, .5]], [0, -1, 0], c);
      if (!solid(i + 1, j)) add([[x0, y1, -.5], [x0, y1, .5], [x1, y1, .5], [x1, y1, -.5]], [0, 1, 0], c);
    }
    const pt = (x, y, z = 0) => [x - cx, y - cy, z];
    const L = [[0, 0], [nx, 0], [nx, ny - cut], [nx - cut, ny - cut], [nx - cut, ny], [0, ny]];   // the design space
    Object.assign(m, {nx, ny, quads, radius: 0.5 * Math.hypot(nx, ny, 1),   // render_paraview.py: the bounding sphere
                      outline: [-.5, .5].map(z => L.map(([x, y]) => pt(x, y, z))), marks: null});
    const cfg = data.config;
    if (!cfg || !nx || !(cfg.lx_m > 0) || !Array.isArray(cfg.force_n)) return;
    const size = Math.max(nx, ny), arm = nx - cut;
    // Clamp: the top edge of the vertical arm (y = ny, x from 0 to nx - cut), hatched on the outside.
    const n = Math.max(3, Math.min(24, Math.round(arm / (0.05 * size)))), tick = 0.045 * size / Math.hypot(.5, 1, .5);
    const ticks = Array.from({length: n + 1}, (_, k) => {
      const x = k * arm / n;
      return [pt(x, ny), pt(x - .5 * tick, ny + tick, .5 * tick)];   // outward (+y), tilted in x and z: seen from the front and from above
    });
    // Load: the tip band on the right edge, from ny - cut - patch to ny - cut (levelset.py, main.py MNBC).
    const patch = Math.round(nx * (cfg.load_patch_m || 0) / cfg.lx_m), [fx, fy] = cfg.force_n.map(Number);
    const mag = Math.hypot(fx, fy), dir = mag ? [fx / mag, fy / mag, 0] : [0, 0, 0], len = 0.2 * size;
    const yc = ny - cut - patch / 2;
    // The arrow ends on the band (a push) unless its tail would then sit in material; then it starts there (a pull).
    // The right edge counts as material, so a load along it pulls from the band into the cut corner instead.
    const inside = (x, y) => x > 0 && x <= nx && y > 0 && y < ny && !(x > nx - cut && y > ny - cut);
    const push = !inside(nx - dir[0] * len / 2, yc - dir[1] * len / 2);
    const tail = push ? pt(nx - dir[0] * len, yc - dir[1] * len) : pt(nx, yc);
    const head = push ? pt(nx, yc) : pt(nx + dir[0] * len, yc + dir[1] * len);
    m.marks = {clamp: [pt(0, ny), pt(arm, ny)], ticks, band: [pt(nx, ny - cut - patch), pt(nx, ny - cut)],
               point: pt(nx, yc), dir, tail, head, far: push ? tail : head, near: push ? head : tail,
               label: `F = (${fmt(fx)}, ${fmt(fy)}) N`, force: mag > 0};
  }

  function readTheme() {
    const colour = value => { probe.style.color = value; return getComputedStyle(probe).color; };
    const rgb = s => { const v = (s.match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number);
                       return s.startsWith("color(") ? v.map(c => c * 255) : v; };
    const bg = rgb(colour("var(--st-background-color, #ffffff)")), ink = rgb(colour("var(--st-text-color, #1f2328)"));
    // No light/dark flag reaches components: dark when the background's relative luminance is below 0.179.
    const lin = bg.map(c => { const u = c / 255; return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4; });
    const dark = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.179;
    const rgba = (c, a) => `rgba(${c.map(Math.round).join(",")},${a})`;
    theme = {dark, ink: rgba(ink, 1), ghost: rgba(ink, 0.35), halo: rgba(ink, 0.5), paper: rgba(bg, 0.85),
             load: colour(`var(--st-orange-text-color, ${dark ? "#ffbd45" : "#e2660c"})`),
             font: getComputedStyle(wrap).fontFamily};
    wrap.classList.toggle("dark", dark);
  }

  function update(data) {
    build(data);
    readTheme();
    title.textContent = title.title = data.title || "";
    marksToggle.hidden = !m.marks;
    canvas.setAttribute("aria-label", `${data.title || "Design"}: ${m.nx} × ${m.ny} elements, one cube per element with material`);
    draw();
  }

  function draw() {
    const w = stage.clientWidth, h = stage.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const {r, u, d} = camera(view.az, view.el);
    const s = Math.min(w, h) / (2.1 * m.radius) * view.zoom;   // render_paraview.py: parallel scale 1.05 x radius
    const P = p => [w / 2 + s * dot(p, r), h / 2 - s * dot(p, u)];
    const path = (pts, close = true) => {
      pts.forEach((p, k) => { const [x, y] = P(p); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      if (close) ctx.closePath();
    };
    if (view.ghost) {
      ctx.strokeStyle = theme.ghost; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
      for (const loop of m.outline) { ctx.beginPath(); path(loop); ctx.stroke(); }
      ctx.setLineDash([]);
    }
    const faces = [];
    for (const q of m.quads) {
      const facing = dot(q.n, d);
      if (facing <= 1e-9) continue;   // back-face culling
      faces.push({q, depth: dot(q.m, d), shade: Math.max(0.25, dot(q.n, r) * LIGHT[0] + dot(q.n, u) * LIGHT[1] + facing * LIGHT[2])});
    }
    faces.sort((a, b) => a.depth - b.depth);   // painter's order, far to near
    if (theme.dark) {   // a light rim around the silhouette: the fills below cover it everywhere inside
      ctx.beginPath();
      for (const f of faces) path(f.q.p);
      ctx.strokeStyle = theme.halo; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.stroke();
    }
    ctx.lineWidth = 0.5; ctx.strokeStyle = "rgba(20,35,55,0.18)";
    for (const f of faces) {
      const k = 0.62 + 0.38 * f.shade;
      ctx.fillStyle = `rgb(${f.q.c.map(c => Math.min(255, Math.round(c * k + 22))).join(",")})`;
      ctx.beginPath(); path(f.q.p); ctx.fill(); ctx.stroke();
    }
    if (view.marks && m.marks) marks(ctx, P, d, w, h);
    const name = current();
    if (name !== shown) { shown = name; hud(); }
    at.innerHTML = name ? `<b>${hudTitle(name)}</b> · ${HUD_LOOK[name]} · orthographic`
      : `Orbit · azimuth ${Math.round(wrapAz(view.az))}° · elevation ${Math.round(view.el)}° · orthographic`;
  }

  function marks(ctx, P, d, w, h) {
    const k = m.marks, line = (a, b) => { const [x0, y0] = P(a), [x1, y1] = P(b);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
    ctx.lineCap = "round";
    ctx.strokeStyle = theme.ink; ctx.lineWidth = 2.5; line(...k.clamp);
    ctx.lineWidth = 1.2; for (const t of k.ticks) line(...t);
    ctx.strokeStyle = ctx.fillStyle = theme.load; ctx.lineWidth = 4; line(...k.band);
    if (!k.force) return;
    const [x0, y0] = P(k.tail), [x1, y1] = P(k.head), [px, py] = P(k.point), len = Math.hypot(x1 - x0, y1 - y0);
    let lx, ly, ux, uy;
    if (len > 10) {
      const ex = (x1 - x0) / len, ey = (y1 - y0) / len, hl = Math.min(11, len / 2);
      ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ex * hl * 0.8, y1 - ey * hl * 0.8); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - ex * hl - ey * hl * 0.45, y1 - ey * hl + ex * hl * 0.45);
      ctx.lineTo(x1 - ex * hl + ey * hl * 0.45, y1 - ey * hl - ex * hl * 0.45); ctx.closePath(); ctx.fill();
      const [fx, fy] = P(k.far), [nx, ny] = P(k.near);
      ux = (fx - nx) / len; uy = (fy - ny) / len; lx = fx + ux * 8; ly = fy + uy * 8;
    } else {   // along the line of sight: a dot when the force points at the viewer, a cross when away
      ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(px, py, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath();
      if (dot(k.dir, d) > 0) { ctx.arc(px, py, 2.2, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.moveTo(px - 4.5, py - 4.5); ctx.lineTo(px + 4.5, py + 4.5); ctx.moveTo(px + 4.5, py - 4.5); ctx.lineTo(px - 4.5, py + 4.5); ctx.stroke(); }
      ux = 1; uy = 0; lx = px + 12; ly = py;
    }
    // Beyond the free end of the arrow, kept inside the stage.
    ctx.font = `600 12px ${theme.font}`;
    const tw = ctx.measureText(k.label).width;
    const left = lx - (ux > 0.3 ? 0 : ux < -0.3 ? tw : tw / 2), fit = Math.max(6, Math.min(w - tw - 6, left));
    lx = fit;   // pushed back over a sideways arrow at a narrow stage: set it above the arrow instead
    ly = Math.max(8, Math.min(h - 8, ly + (uy > 0.3 ? 7 : uy < -0.3 ? -7 : 0) - (Math.abs(fit - left) > 4 ? 14 : 0)));
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.strokeStyle = theme.paper;
    ctx.strokeText(k.label, lx, ly); ctx.fillText(k.label, lx, ly);
  }

  // Snap to a standard view (and the shared scale) with a short turn, the shorter way round. The timer
  // completes it where animation frames do not run (a hidden tab).
  function snap(name) {
    const [az, el] = STANDARD[name], a0 = view.az, e0 = view.el, z0 = view.zoom, t0 = performance.now();
    const da = ((az - a0) % 360 + 540) % 360 - 180;
    const finish = () => { stop(); Object.assign(view, {az, el, zoom: 1}); draw(); };
    const step = now => {
      const t = (now - t0) / 220, e = t * (2 - t);
      if (t >= 1) return finish();
      Object.assign(view, {az: wrapAz(a0 + da * e), el: e0 + (el - e0) * e, zoom: z0 + (1 - z0) * e});
      draw();
      anim = requestAnimationFrame(step);
    };
    stop();
    anim = requestAnimationFrame(step); timer = setTimeout(finish, 300);
  }
  const stop = () => { cancelAnimationFrame(anim); clearTimeout(timer); };

  // Pointer capture keeps a drag on the canvas, so no listener outlives the viewer.
  canvas.addEventListener("pointerdown", ev => {
    if (ev.button !== 0) return;
    stop();
    canvas.setPointerCapture(ev.pointerId);
    drag = {id: ev.pointerId, x: ev.clientX, y: ev.clientY, az: view.az, el: view.el};
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", ev => {
    if (!drag || ev.pointerId !== drag.id) return;
    view.az = wrapAz(drag.az - (ev.clientX - drag.x) * 0.45);
    view.el = Math.max(-90, Math.min(90, drag.el + (ev.clientY - drag.y) * 0.45));
    draw();
  });
  const release = ev => { if (drag && ev.pointerId === drag.id) { drag = null; canvas.classList.remove("dragging"); } };
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(type, release);
  canvas.addEventListener("wheel", ev => {
    const zoom = Math.max(0.5, Math.min(8, view.zoom * Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015))));
    if (zoom === view.zoom) return;   // at a bound: the page scrolls instead
    ev.preventDefault(); stop(); view.zoom = zoom; draw();
  }, {passive: false});
  canvas.addEventListener("dblclick", () => snap("isometric"));
  const resize = new ResizeObserver(() => draw());
  resize.observe(stage);
  const viewer = {root, update, dispose: () => {
    resize.disconnect(); stop();
    if (view.viewer === viewer) view.viewer = null;
  }};
  return viewer;
}
"""

_JS = view_hud.JS + f"const CSS = {json.dumps(view_hud.CSS + _CSS)};\n" + _MODULE


def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_result_view", js=_JS)(**kwargs)


def render(density: list[list[float]], *, title: str, cut_cells: int, key: str, config: dict | None = None) -> None:
    """density: element area fractions, rows bottom to top (y), columns left to right (x).

    config: the solver config (LBracket2DConfig.to_dict()); with it the clamp and the load are drawn.
    key: the camera and the toggles are kept per key.
    """
    marks = {name: config[name] for name in _OVERLAY_FIELDS if name in config} if config else None
    _component(data={"density": density, "title": title, "cut_cells": cut_cells, "key": key, "config": marks},
               key=f"result_{key}", height=540)
