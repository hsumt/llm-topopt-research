"""Orbitable 3D view of a 2D level-set result: each solid element extruded to unit thickness.

The solver is plane stress; the extrusion (one element deep) is only for viewing.
Drag to orbit, scroll to zoom, double-click to reset. Rendered in the browser.
"""
import streamlit as st

_JS = r"""
const states = {};

export default function (component) {
  const { data, parentElement } = component;
  const key = data.key || "result";
  const view = states[key] || (states[key] = {yaw: -0.65, pitch: 0.75, zoom: 1, ghost: true});
  parentElement.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:relative;height:520px;border:1px solid #d6d6d6;border-radius:10px;overflow:hidden;" +
    "background:radial-gradient(circle at 50% 35%, #ffffff, #e9edf2);font-family:'Source Sans 3',system-ui,sans-serif;";
  parentElement.appendChild(wrap);
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;cursor:grab;display:block;";
  wrap.appendChild(canvas);
  const bar = document.createElement("div");
  bar.style.cssText = "position:absolute;top:10px;left:10px;display:flex;gap:6px;align-items:center;font-size:13px;color:#1f2328;";
  bar.innerHTML = `<span style="background:#ffffffd9;padding:3px 8px;border-radius:6px">${data.title || ""}</span>
    <label style="background:#ffffffd9;padding:3px 8px;border-radius:6px"><input type="checkbox" ${view.ghost ? "checked" : ""}> design-space outline</label>
    <span style="background:#ffffffd9;padding:3px 8px;border-radius:6px">element density
      <span style="display:inline-block;width:70px;height:10px;vertical-align:middle;margin:0 4px;
        background:linear-gradient(90deg,#d6dee8,#26548c);border:1px solid #9aa7b5"></span>0 → 1</span>`;
  wrap.appendChild(bar);
  const hint = document.createElement("div");
  hint.style.cssText = "position:absolute;bottom:8px;left:10px;font-size:12px;color:#555;background:#ffffffc9;padding:2px 8px;border-radius:6px";
  hint.textContent = "Drag to orbit · scroll to zoom · double-click to reset · one cube per element, unit thickness for viewing (the solver is plane stress)";
  wrap.appendChild(hint);

  const grid = data.density || [];
  const ny = grid.length, nx = ny ? grid[0].length : 0;
  const depth = 1;  // one element deep
  const quads = [];
  // Every element with density above 0.001 is drawn as a cube (no 0.5 cut-off, no smoothing).
  const solid = (i, j) => i >= 0 && j >= 0 && i < ny && j < nx && grid[i][j] > 0.001;
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    if (!solid(i, j)) continue;
    const d = grid[i][j], x0 = j, x1 = j + 1, y0 = i, y1 = i + 1;
    quads.push({p: [[x0, y0, depth], [x1, y0, depth], [x1, y1, depth], [x0, y1, depth]], n: [0, 0, 1], d});
    quads.push({p: [[x0, y0, 0], [x0, y1, 0], [x1, y1, 0], [x1, y0, 0]], n: [0, 0, -1], d});
    if (!solid(i, j - 1)) quads.push({p: [[x0, y0, 0], [x0, y0, depth], [x0, y1, depth], [x0, y1, 0]], n: [-1, 0, 0], d});
    if (!solid(i, j + 1)) quads.push({p: [[x1, y0, 0], [x1, y1, 0], [x1, y1, depth], [x1, y0, depth]], n: [1, 0, 0], d});
    if (!solid(i - 1, j)) quads.push({p: [[x0, y0, 0], [x1, y0, 0], [x1, y0, depth], [x0, y0, depth]], n: [0, -1, 0], d});
    if (!solid(i + 1, j)) quads.push({p: [[x0, y1, 0], [x0, y1, depth], [x1, y1, depth], [x1, y1, 0]], n: [0, 1, 0], d});
  }
  // Outline of the L domain (design space), drawn as a ghost frame.
  const cut = data.cut_cells || 0;
  const outline = [[0, 0], [nx, 0], [nx, ny - cut], [nx - cut, ny - cut], [nx - cut, ny], [0, ny]];
  const cx = nx / 2, cy = ny / 2;
  const light = [0.35, -0.45, 0.82];

  function draw() {
    const w = wrap.clientWidth, h = wrap.clientHeight, dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cyaw = Math.cos(view.yaw), syaw = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const scale = 0.8 * Math.min(w, h) / Math.max(nx, ny, 1) * view.zoom;
    const rot = ([x, y, z]) => {
      x -= cx; y -= cy; z -= depth / 2;
      const x1 = cyaw * x - syaw * y, y1 = syaw * x + cyaw * y;   // yaw about z
      const y2 = cp * y1 - sp * z, z2 = sp * y1 + cp * z;          // pitch about x
      return [x1, y2, z2];
    };
    const proj = ([x, y, z]) => [w / 2 + x * scale, h / 2 - y * scale];
    if (view.ghost) {
      ctx.strokeStyle = "#9aa7b5"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
      for (const z of [0, depth]) {
        ctx.beginPath();
        outline.forEach(([x, y], k) => { const [px, py] = proj(rot([x, y, z])); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
        ctx.closePath(); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    const faces = [];
    for (const q of quads) {
      const r = q.p.map(rot);
      const nr = rot([q.n[0] + cx, q.n[1] + cy, q.n[2] + depth / 2]);
      if (nr[2] <= 0) continue;   // back-face culling: rotated normal must face the viewer
      const depthKey = (r[0][2] + r[1][2] + r[2][2] + r[3][2]) / 4;
      faces.push({r, nr, depthKey, d: q.d, top: Math.abs(q.n[2]) === 1});
    }
    faces.sort((a, b) => a.depthKey - b.depthKey);
    for (const f of faces) {
      const shade = Math.max(0.25, f.nr[0] * light[0] + f.nr[1] * light[1] + f.nr[2] * light[2]);
      // Density colour: full elements dark blue, partial (boundary) elements fade toward pale grey.
      const solidRgb = f.top ? [38, 84, 140] : [30, 64, 108], paleRgb = [214, 222, 232];
      const mix = solidRgb.map((c, n) => c * f.d + paleRgb[n] * (1 - f.d));
      const k = 0.62 + 0.38 * shade;
      ctx.fillStyle = `rgb(${mix.map(c => Math.min(255, Math.round(c * k + 22))).join(",")})`;
      ctx.beginPath();
      f.r.forEach((p, k2) => { const [px, py] = proj(p); k2 ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(20,35,55,0.18)"; ctx.lineWidth = 0.5; ctx.stroke();
    }
  }

  let drag = null;
  canvas.addEventListener("mousedown", ev => { drag = {x: ev.clientX, y: ev.clientY, yaw: view.yaw, pitch: view.pitch}; canvas.style.cursor = "grabbing"; });
  window.addEventListener("mousemove", ev => {
    if (!drag) return;
    view.yaw = drag.yaw + (ev.clientX - drag.x) * 0.008;
    view.pitch = Math.max(-1.55, Math.min(1.55, drag.pitch + (ev.clientY - drag.y) * 0.008));
    draw();
  });
  window.addEventListener("mouseup", () => { drag = null; canvas.style.cursor = "grab"; });
  canvas.addEventListener("wheel", ev => { ev.preventDefault(); view.zoom = Math.max(0.3, Math.min(4, view.zoom * (ev.deltaY < 0 ? 1.1 : 0.9))); draw(); }, {passive: false});
  canvas.addEventListener("dblclick", () => { view.yaw = -0.65; view.pitch = 0.75; view.zoom = 1; draw(); });
  bar.querySelector("input").addEventListener("change", ev => { view.ghost = ev.target.checked; draw(); });
  requestAnimationFrame(draw);
}
"""

def _component(**kwargs):
    # Registered on each render: registration belongs to the running app, which tests and reruns may replace.
    return st.components.v2.component("ato_result_view", js=_JS)(**kwargs)


def render(density: list[list[float]], *, title: str, cut_cells: int, key: str) -> None:
    """density: element area fractions, rows bottom to top (y), columns left to right (x)."""
    _component(data={"density": density, "title": title, "cut_cells": cut_cells, "key": key},
               key=f"result_{key}", height=540)
